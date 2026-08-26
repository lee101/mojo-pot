"""C ABI kernels for balanced discrete optimal transport."""

from max.algorithm import parallelize
from std.math import exp, log, sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime HUGE = 1.7976931348623157e308
comptime PARALLEL_MATRIX_THRESHOLD = 1048576


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def min2(a: Float64, b: Float64) -> Float64:
    return a if a < b else b


@export("mpot_emd")
def mpot_emd(
    a_addr: Int,
    b_addr: Int,
    cost_addr: Int,
    plan_addr: Int,
    supply_addr: Int,
    demand_addr: Int,
    distance_addr: Int,
    predecessor_addr: Int,
    potential_addr: Int,
    visited_addr: Int,
    n: Int,
    m: Int,
    max_iter: Int,
    tol: Float64,
) abi("C") -> Int:
    var a = p(a_addr)
    var b = p(b_addr)
    var cost = p(cost_addr)
    var plan = p(plan_addr)
    var supply = p(supply_addr)
    var demand = p(demand_addr)
    var distance = p(distance_addr)
    var predecessor = ip(predecessor_addr)
    var potential = p(potential_addr)
    var visited = ip(visited_addr)
    var nodes = n + m
    comptime W = simdwidthof[DType.float64]()

    var i = 0
    var vector_end = n - n % W
    while i < vector_end:
        supply.store(i, a.load[width=W](i))
        i += W
    while i < n:
        supply[i] = a[i]
        i += 1
    var j = 0
    vector_end = m - m % W
    while j < vector_end:
        demand.store(j, b.load[width=W](j))
        j += W
    while j < m:
        demand[j] = b[j]
        j += 1
    var k = 0
    vector_end = n * m - (n * m) % W
    var zeros = SIMD[DType.float64, W](0.0)
    while k < vector_end:
        plan.store(k, zeros)
        k += W
    while k < n * m:
        plan[k] = 0.0
        k += 1
    for node in range(nodes):
        potential[node] = 0.0

    var minimum_cost = HUGE
    for index in range(n * m):
        if cost[index] < minimum_cost:
            minimum_cost = cost[index]
    var sink_potential = 0.0

    var iteration = 0
    while iteration < max_iter:
        var remaining_vector = SIMD[DType.float64, W](0.0)
        i = 0
        vector_end = n - n % W
        while i < vector_end:
            remaining_vector += supply.load[width=W](i)
            i += W
        var remaining = remaining_vector.reduce_add()
        while i < n:
            remaining += supply[i]
            i += 1
        if remaining <= tol:
            return iteration

        for node in range(nodes):
            distance[node] = HUGE
            predecessor[node] = -2
            visited[node] = 0
        for i in range(n):
            if supply[i] > tol:
                distance[i] = 0.0
                predecessor[i] = -1

        var end_j = -1
        var best_sink_distance = HUGE
        for _ in range(nodes):
            var current = -1
            var current_distance = HUGE
            for candidate_node in range(nodes):
                if (
                    visited[candidate_node] == 0
                    and distance[candidate_node] < current_distance
                ):
                    current_distance = distance[candidate_node]
                    current = candidate_node
            if current < 0 or current_distance >= best_sink_distance:
                break
            visited[current] = 1

            if current >= n:
                var current_j = current - n
                if demand[current_j] > tol:
                    var sink_distance = (
                        current_distance + potential[current] - sink_potential
                    )
                    if sink_distance < best_sink_distance:
                        best_sink_distance = sink_distance
                        end_j = current_j
                for source_i in range(n):
                    var index = source_i * m + current_j
                    if plan[index] > tol:
                        var reduced = (
                            -(cost[index] - minimum_cost)
                            + potential[current]
                            - potential[source_i]
                        )
                        var candidate = current_distance + reduced
                        if candidate < distance[source_i] - 1.0e-15:
                            distance[source_i] = candidate
                            predecessor[source_i] = Int64(current)
            else:
                var target_j = 0
                var target_vector_end = m - m % W
                while target_j < target_vector_end:
                    var target_node = n + target_j
                    var index = current * m + target_j
                    var candidates = (
                        cost.load[width=W](index)
                        - minimum_cost
                        + potential[current]
                        - potential.load[width=W](target_node)
                    )
                    candidates += current_distance
                    var previous_distances = distance.load[width=W](target_node)
                    comptime for lane in range(W):
                        if (
                            candidates[lane]
                            < previous_distances[lane] - 1.0e-15
                        ):
                            predecessor[target_node + lane] = Int64(current)
                    distance.store(
                        target_node, min(previous_distances, candidates)
                    )
                    target_j += W
                while target_j < m:
                    var target_node = n + target_j
                    var index = current * m + target_j
                    var reduced = (
                        cost[index]
                        - minimum_cost
                        + potential[current]
                        - potential[target_node]
                    )
                    var candidate = current_distance + reduced
                    if candidate < distance[target_node] - 1.0e-15:
                        distance[target_node] = candidate
                        predecessor[target_node] = Int64(current)
                    target_j += 1

        if end_j < 0 or best_sink_distance == HUGE:
            return -(iteration + 1)

        var potential_index = 0
        vector_end = nodes - nodes % W
        var sink_distances = SIMD[DType.float64, W](best_sink_distance)
        while potential_index < vector_end:
            potential.store(
                potential_index,
                potential.load[width=W](potential_index)
                + min(
                    distance.load[width=W](potential_index),
                    sink_distances,
                ),
            )
            potential_index += W
        while potential_index < nodes:
            potential[potential_index] += min2(
                distance[potential_index], best_sink_distance
            )
            potential_index += 1
        sink_potential += best_sink_distance

        var node = n + end_j
        var start_i = -1
        var delta = demand[end_j]
        var hops = 0
        while predecessor[node] != -1:
            var previous = Int(predecessor[node])
            if node < n:
                delta = min2(delta, plan[node * m + (previous - n)])
            node = previous
            hops += 1
            if hops > nodes:
                return -(iteration + 1)
        start_i = node
        delta = min2(delta, supply[start_i])
        if delta <= tol:
            return -(iteration + 1)

        node = n + end_j
        while predecessor[node] != -1:
            var previous = Int(predecessor[node])
            if node >= n:
                plan[previous * m + (node - n)] += delta
            else:
                plan[node * m + (previous - n)] -= delta
            node = previous
        supply[start_i] -= delta
        demand[end_j] -= delta
        if supply[start_i] < tol:
            supply[start_i] = 0.0
        if demand[end_j] < tol:
            demand[end_j] = 0.0
        iteration += 1
    var remaining_vector = SIMD[DType.float64, W](0.0)
    i = 0
    vector_end = n - n % W
    while i < vector_end:
        remaining_vector += supply.load[width=W](i)
        i += W
    var remaining = remaining_vector.reduce_add()
    while i < n:
        remaining += supply[i]
        i += 1
    return max_iter if remaining <= tol else -(max_iter + 1)


def kernel_value(cost: Ptr, index: Int, reg: Float64) -> Float64:
    return exp(-cost[index] / reg)


def logsumexp_row(
    cost: Ptr, log_v: Ptr, row: Int, m: Int, reg: Float64
) -> Float64:
    var maximum = -HUGE
    for j in range(m):
        var value = -cost[row * m + j] / reg + log_v[j]
        if value > maximum:
            maximum = value
    var total = 0.0
    for j in range(m):
        total += exp(-cost[row * m + j] / reg + log_v[j] - maximum)
    return maximum + log(total)


def logsumexp_col(
    cost: Ptr, log_u: Ptr, col: Int, n: Int, m: Int, reg: Float64
) -> Float64:
    var maximum = -HUGE
    for i in range(n):
        var value = -cost[i * m + col] / reg + log_u[i]
        if value > maximum:
            maximum = value
    var total = 0.0
    for i in range(n):
        total += exp(-cost[i * m + col] / reg + log_u[i] - maximum)
    return maximum + log(total)


@export("mpot_sinkhorn")
def mpot_sinkhorn(
    a_addr: Int,
    b_addr: Int,
    cost_addr: Int,
    plan_addr: Int,
    u_addr: Int,
    v_addr: Int,
    error_addr: Int,
    kernel_t_addr: Int,
    n: Int,
    m: Int,
    reg: Float64,
    max_iter: Int,
    stop_threshold: Float64,
    log_domain: Int,
    parallel_enabled: Int,
) abi("C") -> Int:
    var a = p(a_addr)
    var b = p(b_addr)
    var cost = p(cost_addr)
    var plan = p(plan_addr)
    var u = p(u_addr)
    var v = p(v_addr)
    var error = p(error_addr)
    error[0] = HUGE
    comptime W = simdwidthof[DType.float64]()
    var use_parallel = (
        parallel_enabled != 0 and n * m >= PARALLEL_MATRIX_THRESHOLD
    )

    if log_domain == 0:
        var kernel_t = p(kernel_t_addr)
        for i in range(n):
            u[i] = 1.0 / Float64(n)
        for j in range(m):
            v[j] = 1.0 / Float64(m)
        var index = 0
        var vector_end = n * m - (n * m) % W
        while index < vector_end:
            plan.store(
                index,
                exp(-cost.load[width=W](index) / reg),
            )
            index += W
        while index < n * m:
            plan[index] = kernel_value(cost, index, reg)
            index += 1
        for i in range(n):
            for j in range(m):
                kernel_t[j * n + i] = plan[i * m + j]

        @parameter
        def update_u(i: Int):
            var totals = SIMD[DType.float64, W](0.0)
            var j = 0
            var row_start = i * m
            var row_end = m - m % W
            while j < row_end:
                totals += plan.load[width=W](row_start + j) * v.load[width=W](j)
                j += W
            var denominator = totals.reduce_add()
            while j < m:
                denominator += plan[row_start + j] * v[j]
                j += 1
            u[i] = a[i] / denominator if denominator > 0.0 else 0.0

        @parameter
        def update_v(j: Int):
            var totals = SIMD[DType.float64, W](0.0)
            var i = 0
            var row_start = j * n
            var row_end = n - n % W
            while i < row_end:
                totals += kernel_t.load[width=W](row_start + i) * u.load[
                    width=W
                ](i)
                i += W
            var denominator = totals.reduce_add()
            while i < n:
                denominator += kernel_t[row_start + i] * u[i]
                i += 1
            v[j] = b[j] / denominator if denominator > 0.0 else 0.0

        @parameter
        def scale_row(i: Int):
            var row_start = i * m
            var j = 0
            var row_end = m - m % W
            var ui = u[i]
            while j < row_end:
                plan.store(
                    row_start + j,
                    plan.load[width=W](row_start + j) * v.load[width=W](j) * ui,
                )
                j += W
            while j < m:
                plan[row_start + j] *= ui * v[j]
                j += 1

        for iteration in range(max_iter):
            if use_parallel:
                parallelize[update_u](n)
            else:
                for i in range(n):
                    update_u(i)
            for i in range(n):
                if u[i] == 0.0:
                    return -(iteration + 1)
            if use_parallel:
                parallelize[update_v](m)
            else:
                for j in range(m):
                    update_v(j)
            for j in range(m):
                if v[j] == 0.0:
                    return -(iteration + 1)
            if iteration % 10 == 0:
                var err = 0.0
                for i in range(n):
                    var totals = SIMD[DType.float64, W](0.0)
                    var j = 0
                    var row_start = i * m
                    var row_end = m - m % W
                    while j < row_end:
                        totals += plan.load[width=W](row_start + j) * v.load[
                            width=W
                        ](j)
                        j += W
                    var marginal = totals.reduce_add()
                    while j < m:
                        marginal += plan[row_start + j] * v[j]
                        j += 1
                    marginal *= u[i]
                    err += abs(marginal - a[i])
                error[0] = err
                if err <= stop_threshold:
                    if use_parallel:
                        parallelize[scale_row](n)
                    else:
                        for i in range(n):
                            scale_row(i)
                    return iteration + 1
        if use_parallel:
            parallelize[scale_row](n)
        else:
            for i in range(n):
                scale_row(i)
        return max_iter

    for i in range(n):
        u[i] = 0.0
    for j in range(m):
        v[j] = 0.0
    for iteration in range(max_iter):
        for i in range(n):
            if a[i] == 0.0:
                u[i] = -HUGE
            else:
                u[i] = log(a[i]) - logsumexp_row(cost, v, i, m, reg)
        for j in range(m):
            if b[j] == 0.0:
                v[j] = -HUGE
            else:
                v[j] = log(b[j]) - logsumexp_col(cost, u, j, n, m, reg)
        if iteration % 10 == 0:
            var err = 0.0
            for i in range(n):
                if a[i] == 0.0:
                    continue
                var marginal = 0.0
                for j in range(m):
                    if b[j] > 0.0:
                        marginal += exp(u[i] + v[j] - cost[i * m + j] / reg)
                err += abs(marginal - a[i])
            error[0] = err
            if err <= stop_threshold:
                for i in range(n):
                    for j in range(m):
                        plan[i * m + j] = (
                            exp(u[i] + v[j] - cost[i * m + j] / reg) if a[i]
                            > 0.0
                            and b[j] > 0.0 else 0.0
                        )
                return iteration + 1
    for i in range(n):
        for j in range(m):
            plan[i * m + j] = (
                exp(u[i] + v[j] - cost[i * m + j] / reg) if a[i] > 0.0
                and b[j] > 0.0 else 0.0
            )
    return max_iter


@export("mpot_greenkhorn")
def mpot_greenkhorn(
    a_addr: Int,
    b_addr: Int,
    cost_addr: Int,
    plan_addr: Int,
    u_addr: Int,
    v_addr: Int,
    rows_addr: Int,
    cols_addr: Int,
    error_addr: Int,
    n: Int,
    m: Int,
    reg: Float64,
    max_iter: Int,
    stop_threshold: Float64,
) abi("C") -> Int:
    var a = p(a_addr)
    var b = p(b_addr)
    var cost = p(cost_addr)
    var plan = p(plan_addr)
    var u = p(u_addr)
    var v = p(v_addr)
    var rows = p(rows_addr)
    var cols = p(cols_addr)
    var error = p(error_addr)

    for i in range(n):
        u[i] = 1.0 / Float64(n)
        rows[i] = 0.0
    for j in range(m):
        v[j] = 1.0 / Float64(m)
        cols[j] = 0.0
    for i in range(n):
        for j in range(m):
            var value = u[i] * kernel_value(cost, i * m + j, reg) * v[j]
            plan[i * m + j] = value
            rows[i] += value
            cols[j] += value

    for iteration in range(max_iter):
        var best_error = -1.0
        var best_index = 0
        var row_selected = True
        for i in range(n):
            var violation = abs(rows[i] - a[i])
            if violation > best_error:
                best_error = violation
                best_index = i
                row_selected = True
        for j in range(m):
            var violation = abs(cols[j] - b[j])
            if violation > best_error:
                best_error = violation
                best_index = j
                row_selected = False
        error[0] = best_error
        if best_error <= stop_threshold:
            return iteration
        if row_selected:
            if rows[best_index] <= 0.0:
                return -(iteration + 1)
            var factor = a[best_index] / rows[best_index]
            u[best_index] *= factor
            for j in range(m):
                var index = best_index * m + j
                var previous = plan[index]
                plan[index] *= factor
                cols[j] += plan[index] - previous
            rows[best_index] = a[best_index]
        else:
            if cols[best_index] <= 0.0:
                return -(iteration + 1)
            var factor = b[best_index] / cols[best_index]
            v[best_index] *= factor
            for i in range(n):
                var index = i * m + best_index
                var previous = plan[index]
                plan[index] *= factor
                rows[i] += plan[index] - previous
            cols[best_index] = b[best_index]
    return max_iter


@export("mpot_dist")
def mpot_dist(
    x_addr: Int,
    y_addr: Int,
    result_addr: Int,
    n: Int,
    m: Int,
    d: Int,
) abi("C"):
    var x = p(x_addr)
    var y = p(y_addr)
    var result = p(result_addr)
    for i in range(n):
        for j in range(m):
            var total = 0.0
            for k in range(d):
                var difference = x[i * d + k] - y[j * d + k]
                total += difference * difference
            result[i * m + j] = total


@export("mpot_emd_1d")
def mpot_emd_1d(
    sorted_a_addr: Int,
    sorted_b_addr: Int,
    order_a_addr: Int,
    order_b_addr: Int,
    plan_addr: Int,
    path_a_addr: Int,
    path_b_addr: Int,
    n: Int,
    m: Int,
    tol: Float64,
) abi("C") -> Int:
    var sorted_a = p(sorted_a_addr)
    var sorted_b = p(sorted_b_addr)
    var order_a = ip(order_a_addr)
    var order_b = ip(order_b_addr)
    var plan = p(plan_addr)
    var path_a = ip(path_a_addr)
    var path_b = ip(path_b_addr)
    for index in range(n * m):
        plan[index] = 0.0

    var i = 0
    var j = 0
    var count = 0
    while i < n and j < m:
        var source = sorted_a[i]
        var target = sorted_b[j]
        var amount = min2(source, target)
        var original_i = Int(order_a[i])
        var original_j = Int(order_b[j])
        plan[original_i * m + original_j] += amount
        path_a[count] = Int64(original_i)
        path_b[count] = Int64(original_j)
        count += 1
        if source < target - tol:
            sorted_b[j] = target - source
            sorted_a[i] = 0.0
            i += 1
        else:
            sorted_a[i] = source - target
            sorted_b[j] = 0.0
            j += 1
    return count


@export("mpot_wasserstein_1d")
def mpot_wasserstein_1d(
    sorted_x_addr: Int,
    sorted_y_addr: Int,
    sorted_a_addr: Int,
    sorted_b_addr: Int,
    n: Int,
    m: Int,
    exponent: Float64,
) abi("C") -> Float64:
    var sorted_x = p(sorted_x_addr)
    var sorted_y = p(sorted_y_addr)
    var sorted_a = p(sorted_a_addr)
    var sorted_b = p(sorted_b_addr)
    var i = 0
    var j = 0
    var result = 0.0
    while i < n and j < m:
        var source = sorted_a[i]
        var target = sorted_b[j]
        var amount = min2(source, target)
        var distance = abs(sorted_x[i] - sorted_y[j])
        if distance > 0.0:
            var powered = distance
            if exponent == 2.0:
                powered = distance * distance
            elif exponent == 3.0:
                powered = distance * distance * distance
            elif exponent == 4.0:
                var square = distance * distance
                powered = square * square
            elif exponent != 1.0:
                powered = exp(exponent * log(distance))
            result += amount * powered
        if source < target:
            sorted_b[j] = target - source
            i += 1
        else:
            sorted_a[i] = source - target
            j += 1
    return result
