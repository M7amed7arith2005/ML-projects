graph={
    'S':[('A',2),('B',3),('D',5)],
    'A':[('C',4)],
    'B':[('D',4)],
    'C':[('D',1),('G',2)],
    'D':[('G',5)],
    'G':[]
}
# A Helper function => to calc the total cost of the path
def path_cost(path):
    total_cost=0
    for (node,cost) in path:
        total_cost+=cost
    return total_cost
def UCS(graph,start,goal):
    visited=[]
    queue=[[(start,0)]]
    while queue:
        queue.sort(key=path_cost)
        path=queue.pop(0)
        node=path[-1][0]
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        else:
            childrenOfNode=graph.get(node,[])
            for (node2,cost) in childrenOfNode:
                new_path=path.copy()
                new_path.append((node2,cost))
                queue.append(new_path)
    return None
solution=UCS(graph,'S','G')
print(f"The Solution of the the graph Using UCS : {solution}")
