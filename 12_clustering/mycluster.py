
"""
cluster = {"c1" : ["x", "x", "x", "x", "x", 0],
           "c2" : [],
           "c3" : []}



item = [("x", "c1"),
        ("x", "c1",
         ...)]

comb = [(("x", "c1"),("x", "c1")),
        (("o", "c1"), ("d", "c3"))
        ]

TP, TN, FP, FN = [0,0,0,0]

for item in comb: 
    if item[0][0] == item[1][0] and item[0][1] == item[1][1]:
        TP +=1
    if item[0][0] != item[1][0] and item[0][1] == item[1][1]:
        FP +=1
    if item[0][0] == item[1][0] and item[0][1] != item[1][1]:
        FN +=1
    if item[0][0] != item[1][0] and item[0][1] != item[1][1]:
        TN +=1

"""

TP = 20
FP = 20
TN = 72
FN = 24


RI = (TP + TN) / (TP + TN + FP + FN)
print(RI)

P = TP / (TP + FP)
R = TP / (TP + FN)

F1 = (2*P*R) / (P + R)
print(F1)

