def Dfs(graph,start,goal):
    visited=[]
    Stack=[[start]]
    while Stack:
        path=Stack.pop()
        node=path[-1]
        if node is visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        else:
            ChildrenOfNode=graph.get(node,[])
            for node2 in ChildrenOfNode:
                new_path=path.copy()
                new_path.append(node2)
                Stack.append(new_path)
    return None
graph={
    'S':['A','B','D'],
    'A':['C'],
    'B':['D'],
    'C':['D','G'],
    'D':['G'],
    'G':[]
}
solution=Dfs(graph,'S','G')
print(f"The Solution if the path is : {solution}")
