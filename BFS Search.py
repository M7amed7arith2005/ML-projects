def bfs(graph,start,goal):
    visited=[]
    queue=[[start]]
    while queue:
        path=queue.pop(0)
        node=path[-1]
        if node in visited:
            continue
        visited.append(node)
        if node ==goal:
            return path
        else:
            childrenOfNodes=graph.get(node,[])
            for node2 in childrenOfNodes:
                new_path=path.copy()
                new_path.append(node2)
                queue.append(new_path)
    return None
graph={
    'S':['A','B','D'],
    'A':['C'],
    'B':['D'],
    'C':['D','G'],
    'D':['G'],
    'G':[]
}
solution=bfs(graph,'S','G')
print(f"The Solution of the path is : {solution}")
