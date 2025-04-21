class Node:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
class binary_tree:
    def __init__(self,root):
        self.root=Node(root)
    def preorder(self,start,traverse):
        if start:
            traverse+=str(start.value)+" --> "
            traverse=self.preorder(start.left,traverse)
            traverse=self.preorder(start.right,traverse)
        return traverse
    def inorder(self,start,traverse):
        if start:
            traverse=self.inorder(start.left,traverse)
            traverse+=str(start.value) + " --> "
            traverse=self.inorder(start.left,traverse)
        return traverse
    def postorder(self,start,traverse):
        if start:
            traverse=self.postorder(start.left,traverse)
            traverse=self.postorder(start.right,traverse)
            traverse+=str(start.value)+ " --> "
        return traverse
    def printtree(self,traverse_value):
        if traverse_value=="preorder":
            return self.preorder(tree.root,"")
        elif traverse_value=="inorder":
            return self.inorder(tree.root,"")
        elif traverse_value=="postorder":
            return self.postorder(tree.root,"")
        else:
            return "You Enter a Wrong string"
mylist=[10,22,45,56,90,60,68]
tree=binary_tree(mylist[0])
tree.root.left=Node(mylist[1])
tree.root.left.left=Node(mylist[2])
tree.root.left.right=Node(mylist[3])
tree.root.right=Node(mylist[4])
tree.root.right.left=Node(mylist[5])
tree.root.right.right=Node(mylist[6])
choice=int(input("Choose from the list :\n1-'Pre-order'\n2-'In-order'\n3-'Post-order'"))
if choice==1:
    print(tree.printtree("preorder"))
elif choice==2:
    print(tree.printtree("inorder"))
elif choice==3:
    print(tree.printtree("postorder"))
else:
    print("You Enter Un-supported number...!")
