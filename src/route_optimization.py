from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_demo_route():
    distance_matrix = [
        [0, 10, 15, 20, 12],
        [10, 0, 9, 14, 8],
        [15, 9, 0, 11, 7],
        [20, 14, 11, 0, 13],
        [12, 8, 7, 13, 0],
    ]
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 2, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(params)

    routes = []
    if solution:
        for vehicle in range(2):
            index = routing.Start(vehicle)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            routes.append(route)
    return routes
