"""C ABI kernels for balanced discrete optimal transport."""

from std.math import exp, log, sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime HUGE = 1.7976931348623157e308


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
    heap_addr: Int,
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
    var heap = ip(heap_addr)
    var nodes = n + m
    comptime W = simdwidthof[DType.float64]()

    var i = 0
    var vector_end = n - n % W
    while i < vector_end:
        supply.unsafe_store(i, a.unsafe_load[width=W](i))
        i += W
    while i < n:
        supply[unsafe_offset=i] = a[unsafe_offset=i]
        i += 1
    var j = 0
    vector_end = m - m % W
    while j < vector_end:
        demand.unsafe_store(j, b.unsafe_load[width=W](j))
        j += W
    while j < m:
        demand[unsafe_offset=j] = b[unsafe_offset=j]
        j += 1
    var k = 0
    vector_end = n * m - (n * m) % W
    var zeros = SIMD[DType.float64, W](0.0)
    while k < vector_end:
        plan.unsafe_store(k, zeros)
        k += W
    while k < n * m:
        plan[unsafe_offset=k] = 0.0
        k += 1
    for node in range(nodes):
        potential[unsafe_offset=node] = 0.0

    var minimum_cost = HUGE
    for index in range(n * m):
        if cost[unsafe_offset=index] < minimum_cost:
            minimum_cost = cost[unsafe_offset=index]
    var sink_potential = 0.0

    var iteration = 0
    while iteration < max_iter:
        var remaining_vector = SIMD[DType.float64, W](0.0)
        i = 0
        vector_end = n - n % W
        while i < vector_end:
            remaining_vector += supply.unsafe_load[width=W](i)
            i += W
        var remaining = remaining_vector.reduce_add()
        while i < n:
            remaining += supply[unsafe_offset=i]
            i += 1
        if remaining <= tol:
            return iteration

        var heap_size = 0
        for node in range(nodes):
            distance[unsafe_offset=node] = HUGE
            predecessor[unsafe_offset=node] = -2
            visited[unsafe_offset=node] = -1
        for i in range(n):
            if supply[unsafe_offset=i] > tol:
                distance[unsafe_offset=i] = 0.0
                predecessor[unsafe_offset=i] = -1
                heap[unsafe_offset=heap_size] = Int64(i)
                visited[unsafe_offset=i] = Int64(heap_size)
                heap_size += 1

        var end_j = -1
        var best_sink_distance = HUGE
        while heap_size > 0:
            var current = Int(heap[unsafe_offset=0])
            var current_distance = distance[unsafe_offset=current]
            heap_size -= 1
            visited[unsafe_offset=current] = -2
            if heap_size > 0:
                var replacement = Int(heap[unsafe_offset=heap_size])
                heap[unsafe_offset=0] = Int64(replacement)
                visited[unsafe_offset=replacement] = 0
                var heap_index = 0
                while True:
                    var left = 2 * heap_index + 1
                    if left >= heap_size:
                        break
                    var right = left + 1
                    var child = left
                    if (
                        right < heap_size
                        and distance[unsafe_offset=Int(heap[unsafe_offset=right])]
                        < distance[unsafe_offset=Int(heap[unsafe_offset=left])]
                    ):
                        child = right
                    if distance[unsafe_offset=replacement] <= distance[unsafe_offset=Int(heap[unsafe_offset=child])]:
                        break
                    var child_node = Int(heap[unsafe_offset=child])
                    heap[unsafe_offset=heap_index] = Int64(child_node)
                    visited[unsafe_offset=child_node] = Int64(heap_index)
                    heap_index = child
                heap[unsafe_offset=heap_index] = Int64(replacement)
                visited[unsafe_offset=replacement] = Int64(heap_index)
            if current_distance >= best_sink_distance:
                break

            if current >= n:
                var current_j = current - n
                if demand[unsafe_offset=current_j] > tol:
                    var sink_distance = (
                        current_distance + potential[unsafe_offset=current] - sink_potential
                    )
                    if sink_distance < best_sink_distance:
                        best_sink_distance = sink_distance
                        end_j = current_j
                for source_i in range(n):
                    var index = source_i * m + current_j
                    if plan[unsafe_offset=index] > tol:
                        var reduced = (
                            -(cost[unsafe_offset=index] - minimum_cost)
                            + potential[unsafe_offset=current]
                            - potential[unsafe_offset=source_i]
                        )
                        var candidate = current_distance + reduced
                        if (
                            visited[unsafe_offset=source_i] != -2
                            and candidate < distance[unsafe_offset=source_i] - 1.0e-15
                        ):
                            distance[unsafe_offset=source_i] = candidate
                            predecessor[unsafe_offset=source_i] = Int64(current)
                            var position = Int(visited[unsafe_offset=source_i])
                            if position == -1:
                                position = heap_size
                                heap[unsafe_offset=heap_size] = Int64(source_i)
                                visited[unsafe_offset=source_i] = Int64(position)
                                heap_size += 1
                            while position > 0:
                                var parent = (position - 1) // 2
                                var parent_node = Int(heap[unsafe_offset=parent])
                                if distance[unsafe_offset=parent_node] <= candidate:
                                    break
                                heap[unsafe_offset=position] = Int64(parent_node)
                                visited[unsafe_offset=parent_node] = Int64(position)
                                position = parent
                            heap[unsafe_offset=position] = Int64(source_i)
                            visited[unsafe_offset=source_i] = Int64(position)
            else:
                var target_j = 0
                var target_vector_end = m - m % W
                while target_j < target_vector_end:
                    var target_node = n + target_j
                    var index = current * m + target_j
                    var candidates = (
                        cost.unsafe_load[width=W](index)
                        - minimum_cost
                        + potential[unsafe_offset=current]
                        - potential.unsafe_load[width=W](target_node)
                    )
                    candidates += current_distance
                    var previous_distances = distance.unsafe_load[width=W](target_node)
                    comptime for lane in range(W):
                        if (
                            visited[unsafe_offset=target_node + lane] != -2
                            and candidates[lane]
                            < previous_distances[lane] - 1.0e-15
                        ):
                            var updated_node = target_node + lane
                            distance[unsafe_offset=updated_node] = candidates[lane]
                            predecessor[unsafe_offset=updated_node] = Int64(current)
                            var position = Int(visited[unsafe_offset=updated_node])
                            if position == -1:
                                position = heap_size
                                heap[unsafe_offset=heap_size] = Int64(updated_node)
                                visited[unsafe_offset=updated_node] = Int64(position)
                                heap_size += 1
                            while position > 0:
                                var parent = (position - 1) // 2
                                var parent_node = Int(heap[unsafe_offset=parent])
                                if distance[unsafe_offset=parent_node] <= candidates[lane]:
                                    break
                                heap[unsafe_offset=position] = Int64(parent_node)
                                visited[unsafe_offset=parent_node] = Int64(position)
                                position = parent
                            heap[unsafe_offset=position] = Int64(updated_node)
                            visited[unsafe_offset=updated_node] = Int64(position)
                    target_j += W
                while target_j < m:
                    var target_node = n + target_j
                    var index = current * m + target_j
                    var reduced = (
                        cost[unsafe_offset=index]
                        - minimum_cost
                        + potential[unsafe_offset=current]
                        - potential[unsafe_offset=target_node]
                    )
                    var candidate = current_distance + reduced
                    if (
                        visited[unsafe_offset=target_node] != -2
                        and candidate < distance[unsafe_offset=target_node] - 1.0e-15
                    ):
                        distance[unsafe_offset=target_node] = candidate
                        predecessor[unsafe_offset=target_node] = Int64(current)
                        var position = Int(visited[unsafe_offset=target_node])
                        if position == -1:
                            position = heap_size
                            heap[unsafe_offset=heap_size] = Int64(target_node)
                            visited[unsafe_offset=target_node] = Int64(position)
                            heap_size += 1
                        while position > 0:
                            var parent = (position - 1) // 2
                            var parent_node = Int(heap[unsafe_offset=parent])
                            if distance[unsafe_offset=parent_node] <= candidate:
                                break
                            heap[unsafe_offset=position] = Int64(parent_node)
                            visited[unsafe_offset=parent_node] = Int64(position)
                            position = parent
                        heap[unsafe_offset=position] = Int64(target_node)
                        visited[unsafe_offset=target_node] = Int64(position)
                    target_j += 1

        if end_j < 0 or best_sink_distance == HUGE:
            return -(iteration + 1)

        var potential_index = 0
        vector_end = nodes - nodes % W
        var sink_distances = SIMD[DType.float64, W](best_sink_distance)
        while potential_index < vector_end:
            potential.unsafe_store(
                potential_index,
                potential.unsafe_load[width=W](potential_index)
                + min(
                    distance.unsafe_load[width=W](potential_index),
                    sink_distances,
                ),
            )
            potential_index += W
        while potential_index < nodes:
            potential[unsafe_offset=potential_index] += min2(
                distance[unsafe_offset=potential_index], best_sink_distance
            )
            potential_index += 1
        sink_potential += best_sink_distance

        var node = n + end_j
        var start_i = -1
        var delta = demand[unsafe_offset=end_j]
        var hops = 0
        while predecessor[unsafe_offset=node] != -1:
            var previous = Int(predecessor[unsafe_offset=node])
            if node < n:
                delta = min2(delta, plan[unsafe_offset=node * m + (previous - n)])
            node = previous
            hops += 1
            if hops > nodes:
                return -(iteration + 1)
        start_i = node
        delta = min2(delta, supply[unsafe_offset=start_i])
        if delta <= tol:
            return -(iteration + 1)

        node = n + end_j
        while predecessor[unsafe_offset=node] != -1:
            var previous = Int(predecessor[unsafe_offset=node])
            if node >= n:
                plan[unsafe_offset=previous * m + (node - n)] += delta
            else:
                plan[unsafe_offset=node * m + (previous - n)] -= delta
            node = previous
        supply[unsafe_offset=start_i] -= delta
        demand[unsafe_offset=end_j] -= delta
        if supply[unsafe_offset=start_i] < tol:
            supply[unsafe_offset=start_i] = 0.0
        if demand[unsafe_offset=end_j] < tol:
            demand[unsafe_offset=end_j] = 0.0
        iteration += 1
    var remaining_vector = SIMD[DType.float64, W](0.0)
    i = 0
    vector_end = n - n % W
    while i < vector_end:
        remaining_vector += supply.unsafe_load[width=W](i)
        i += W
    var remaining = remaining_vector.reduce_add()
    while i < n:
        remaining += supply[unsafe_offset=i]
        i += 1
    return max_iter if remaining <= tol else -(max_iter + 1)


def kernel_value(cost: Ptr, index: Int, reg: Float64) -> Float64:
    return exp(-cost[unsafe_offset=index] / reg)


def logsumexp_row(
    cost: Ptr, log_v: Ptr, row: Int, m: Int, reg: Float64
) -> Float64:
    var maximum = -HUGE
    for j in range(m):
        var value = -cost[unsafe_offset=row * m + j] / reg + log_v[unsafe_offset=j]
        if value > maximum:
            maximum = value
    var total = 0.0
    for j in range(m):
        total += exp(-cost[unsafe_offset=row * m + j] / reg + log_v[unsafe_offset=j] - maximum)
    return maximum + log(total)


def logsumexp_col(
    cost: Ptr, log_u: Ptr, col: Int, n: Int, m: Int, reg: Float64
) -> Float64:
    var maximum = -HUGE
    for i in range(n):
        var value = -cost[unsafe_offset=i * m + col] / reg + log_u[unsafe_offset=i]
        if value > maximum:
            maximum = value
    var total = 0.0
    for i in range(n):
        total += exp(-cost[unsafe_offset=i * m + col] / reg + log_u[unsafe_offset=i] - maximum)
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
) abi("C") -> Int:
    var a = p(a_addr)
    var b = p(b_addr)
    var cost = p(cost_addr)
    var plan = p(plan_addr)
    var u = p(u_addr)
    var v = p(v_addr)
    var error = p(error_addr)
    error[unsafe_offset=0] = HUGE
    comptime W = simdwidthof[DType.float64]()


    if log_domain == 0:
        var kernel_t = p(kernel_t_addr)
        for i in range(n):
            u[unsafe_offset=i] = 1.0 / Float64(n)
        for j in range(m):
            v[unsafe_offset=j] = 1.0 / Float64(m)
        var index = 0
        var vector_end = n * m - (n * m) % W
        while index < vector_end:
            plan.unsafe_store(
                index,
                exp(-cost.unsafe_load[width=W](index) / reg),
            )
            index += W
        while index < n * m:
            plan[unsafe_offset=index] = kernel_value(cost, index, reg)
            index += 1
        for i in range(n):
            for j in range(m):
                kernel_t[unsafe_offset=j * n + i] = plan[unsafe_offset=i * m + j]

        @__parameter
        def update_u(i: Int):
            var totals = SIMD[DType.float64, W](0.0)
            var j = 0
            var row_start = i * m
            var row_end = m - m % W
            while j < row_end:
                totals += plan.unsafe_load[width=W](row_start + j) * v.unsafe_load[width=W](j)
                j += W
            var denominator = totals.reduce_add()
            while j < m:
                denominator += plan[unsafe_offset=row_start + j] * v[unsafe_offset=j]
                j += 1
            u[unsafe_offset=i] = a[unsafe_offset=i] / denominator if denominator > 0.0 else 0.0

        @__parameter
        def update_v(j: Int):
            var totals = SIMD[DType.float64, W](0.0)
            var i = 0
            var row_start = j * n
            var row_end = n - n % W
            while i < row_end:
                totals += kernel_t.unsafe_load[width=W](row_start + i) * u.unsafe_load[
                    width=W
                ](i)
                i += W
            var denominator = totals.reduce_add()
            while i < n:
                denominator += kernel_t[unsafe_offset=row_start + i] * u[unsafe_offset=i]
                i += 1
            v[unsafe_offset=j] = b[unsafe_offset=j] / denominator if denominator > 0.0 else 0.0

        @__parameter
        def scale_row(i: Int):
            var row_start = i * m
            var j = 0
            var row_end = m - m % W
            var ui = u[unsafe_offset=i]
            while j < row_end:
                plan.unsafe_store(
                    row_start + j,
                    plan.unsafe_load[width=W](row_start + j) * v.unsafe_load[width=W](j) * ui,
                )
                j += W
            while j < m:
                plan[unsafe_offset=row_start + j] *= ui * v[unsafe_offset=j]
                j += 1

        for iteration in range(max_iter):
            # Row and column scaling passes each stream the whole n*m plan
            # matrix once, so they are bandwidth bound; 1.2.0 cannot hand a
            # closure to a worker pool and threading does not pay here.
            for i in range(n):
                update_u(i)
            for i in range(n):
                if u[unsafe_offset=i] == 0.0:
                    return -(iteration + 1)
            for j in range(m):
                update_v(j)
            for j in range(m):
                if v[unsafe_offset=j] == 0.0:
                    return -(iteration + 1)
            if iteration % 10 == 0:
                var err = 0.0
                for i in range(n):
                    var totals = SIMD[DType.float64, W](0.0)
                    var j = 0
                    var row_start = i * m
                    var row_end = m - m % W
                    while j < row_end:
                        totals += plan.unsafe_load[width=W](row_start + j) * v.unsafe_load[
                            width=W
                        ](j)
                        j += W
                    var marginal = totals.reduce_add()
                    while j < m:
                        marginal += plan[unsafe_offset=row_start + j] * v[unsafe_offset=j]
                        j += 1
                    marginal *= u[unsafe_offset=i]
                    err += abs(marginal - a[unsafe_offset=i])
                error[unsafe_offset=0] = err
                if err <= stop_threshold:
                    for i in range(n):
                        scale_row(i)
                    return iteration + 1
            if iteration % 10 == 0:
                var err = 0.0
                for i in range(n):
                    var totals = SIMD[DType.float64, W](0.0)
                    var j = 0
                    var row_start = i * m
                    var row_end = m - m % W
                    while j < row_end:
                        totals += plan.unsafe_load[width=W](row_start + j) * v.unsafe_load[
                            width=W
                        ](j)
                        j += W
                    var marginal = totals.reduce_add()
                    while j < m:
                        marginal += plan[unsafe_offset=row_start + j] * v[unsafe_offset=j]
                        j += 1
                    marginal *= u[unsafe_offset=i]
                    err += abs(marginal - a[unsafe_offset=i])
                error[unsafe_offset=0] = err
                if err <= stop_threshold:
                    for i in range(n):
                        scale_row(i)
                    return iteration + 1
        for i in range(n):
            scale_row(i)
        return max_iter

    for i in range(n):
        u[unsafe_offset=i] = 0.0
    for j in range(m):
        v[unsafe_offset=j] = 0.0
    for iteration in range(max_iter):
        for i in range(n):
            if a[unsafe_offset=i] == 0.0:
                u[unsafe_offset=i] = -HUGE
            else:
                u[unsafe_offset=i] = log(a[unsafe_offset=i]) - logsumexp_row(cost, v, i, m, reg)
        for j in range(m):
            if b[unsafe_offset=j] == 0.0:
                v[unsafe_offset=j] = -HUGE
            else:
                v[unsafe_offset=j] = log(b[unsafe_offset=j]) - logsumexp_col(cost, u, j, n, m, reg)
        if iteration % 10 == 0:
            var err = 0.0
            for i in range(n):
                if a[unsafe_offset=i] == 0.0:
                    continue
                var marginal = 0.0
                for j in range(m):
                    if b[unsafe_offset=j] > 0.0:
                        marginal += exp(u[unsafe_offset=i] + v[unsafe_offset=j] - cost[unsafe_offset=i * m + j] / reg)
                err += abs(marginal - a[unsafe_offset=i])
            error[unsafe_offset=0] = err
            if err <= stop_threshold:
                for i in range(n):
                    for j in range(m):
                        plan[unsafe_offset=i * m + j] = (
                            exp(u[unsafe_offset=i] + v[unsafe_offset=j] - cost[unsafe_offset=i * m + j] / reg) if a[unsafe_offset=i]
                            > 0.0
                            and b[unsafe_offset=j] > 0.0 else 0.0
                        )
                return iteration + 1
    for i in range(n):
        for j in range(m):
            plan[unsafe_offset=i * m + j] = (
                exp(u[unsafe_offset=i] + v[unsafe_offset=j] - cost[unsafe_offset=i * m + j] / reg) if a[unsafe_offset=i] > 0.0
                and b[unsafe_offset=j] > 0.0 else 0.0
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
        u[unsafe_offset=i] = 1.0 / Float64(n)
        rows[unsafe_offset=i] = 0.0
    for j in range(m):
        v[unsafe_offset=j] = 1.0 / Float64(m)
        cols[unsafe_offset=j] = 0.0
    for i in range(n):
        for j in range(m):
            var value = u[unsafe_offset=i] * kernel_value(cost, i * m + j, reg) * v[unsafe_offset=j]
            plan[unsafe_offset=i * m + j] = value
            rows[unsafe_offset=i] += value
            cols[unsafe_offset=j] += value

    for iteration in range(max_iter):
        var best_error = -1.0
        var best_index = 0
        var row_selected = True
        for i in range(n):
            var violation = abs(rows[unsafe_offset=i] - a[unsafe_offset=i])
            if violation > best_error:
                best_error = violation
                best_index = i
                row_selected = True
        for j in range(m):
            var violation = abs(cols[unsafe_offset=j] - b[unsafe_offset=j])
            if violation > best_error:
                best_error = violation
                best_index = j
                row_selected = False
        error[unsafe_offset=0] = best_error
        if best_error <= stop_threshold:
            return iteration
        if row_selected:
            if rows[unsafe_offset=best_index] <= 0.0:
                return -(iteration + 1)
            var factor = a[unsafe_offset=best_index] / rows[unsafe_offset=best_index]
            u[unsafe_offset=best_index] *= factor
            for j in range(m):
                var index = best_index * m + j
                var previous = plan[unsafe_offset=index]
                plan[unsafe_offset=index] *= factor
                cols[unsafe_offset=j] += plan[unsafe_offset=index] - previous
            rows[unsafe_offset=best_index] = a[unsafe_offset=best_index]
        else:
            if cols[unsafe_offset=best_index] <= 0.0:
                return -(iteration + 1)
            var factor = b[unsafe_offset=best_index] / cols[unsafe_offset=best_index]
            v[unsafe_offset=best_index] *= factor
            for i in range(n):
                var index = i * m + best_index
                var previous = plan[unsafe_offset=index]
                plan[unsafe_offset=index] *= factor
                rows[unsafe_offset=i] += plan[unsafe_offset=index] - previous
            cols[unsafe_offset=best_index] = b[unsafe_offset=best_index]
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
    comptime W = simdwidthof[DType.float64]()

    @__parameter
    def compute_row(i: Int):
        for j in range(m):
            var totals = SIMD[DType.float64, W](0.0)
            var k = 0
            var vector_end = d - d % W
            while k < vector_end:
                var differences = (
                    x.unsafe_load[width=W](i * d + k)
                    - y.unsafe_load[width=W](j * d + k)
                )
                totals += differences * differences
                k += W
            var total = totals.reduce_add()
            while k < d:
                var difference = x[unsafe_offset=i * d + k] - y[unsafe_offset=j * d + k]
                total += difference * difference
                k += 1
            result[unsafe_offset=i * m + j] = total

    for i in range(n):
        compute_row(i)


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
        plan[unsafe_offset=index] = 0.0

    var i = 0
    var j = 0
    var count = 0
    while i < n and j < m:
        var source = sorted_a[unsafe_offset=i]
        var target = sorted_b[unsafe_offset=j]
        var amount = min2(source, target)
        var original_i = Int(order_a[unsafe_offset=i])
        var original_j = Int(order_b[unsafe_offset=j])
        plan[unsafe_offset=original_i * m + original_j] += amount
        path_a[unsafe_offset=count] = Int64(original_i)
        path_b[unsafe_offset=count] = Int64(original_j)
        count += 1
        if source < target - tol:
            sorted_b[unsafe_offset=j] = target - source
            sorted_a[unsafe_offset=i] = 0.0
            i += 1
        else:
            sorted_a[unsafe_offset=i] = source - target
            sorted_b[unsafe_offset=j] = 0.0
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
        var source = sorted_a[unsafe_offset=i]
        var target = sorted_b[unsafe_offset=j]
        var amount = min2(source, target)
        var distance = abs(sorted_x[unsafe_offset=i] - sorted_y[unsafe_offset=j])
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
            sorted_b[unsafe_offset=j] = target - source
            i += 1
        else:
            sorted_a[unsafe_offset=i] = source - target
            j += 1
    return result
