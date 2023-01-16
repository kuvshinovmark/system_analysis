from io import StringIO
import csv

def task(csv_string):
    from io import StringIO
    import csv
    import numpy as np
    import math

    def children(mas,max):
        children=[]
        for i in range(1,max+1):
            tmp=[]
            for item in mas:
                if int(item[0])==i:
                    tmp.append(item[1])
            children.append(tmp)
        return children

    def parents(mas,max):
        parents=[]
        for i in range(1,max+1):
            tmp=[]
            for item in mas:
                if int(item[1])==i:
                    tmp.append(item[0])
            parents.append(tmp)
        return parents

    def get_table(table,max,root):
        for k in range(1,max):
            for i in range(table.shape[1]):
                if table[i,root]==k:
                    get_table(table, max,i)
                    for j in range(table.shape[1]):
                        if table[j,i]!=0:
                            table[j,root]=table[j,i]+table[i,root]
        return table

    f = StringIO(csv_string)
    reader = csv.reader(f, delimiter=',')
    out = []
    for row in reader:
        out.append(row)
    max=0

    for item in out:
        if int(item[0])>max:
            max=int(item[0])
        if int(item[1])>max:
            max=int(item[1])
    r1=children(out,max)
    r2=parents(out,max)
    table=np.zeros((max,max))

    for i in range(len(r1)):
        for item in r1[i]:
            table[int(item)-1,i]=1
    root=r2.index([])
    table=get_table(table,max,root)
    r3=[]
    r4=[]

    for j in range(max):
        count=0
        for i in range(max):
            if table[i,j]>1:
                count+=1
        r3.append(count)

    for i in range(max):
        count=0
        for j in range(max):
            if table[i,j]>1:
                count+=1
        r4.append(count)
    r5=[0]*max

    for k in range(1,max):
        count=0
        for i in range(table.shape[1]):
            if table[i,root]==k:
                count+=1
        if count>1:
            for i in range(table.shape[1]):
                if table[i,root]==k:
                    r5[i]=int(count-1)
    for i in range(max):
        r1[i]=len(r1[i])
        r2[i]=len(r2[i])
    l_table=[r1,r2,r3,r4,list(r5)]
    l=np.zeros(max)
    for item in l_table:
        for elem in item:
            l[elem]+=1
    H=0
    for i in range(1,max):
        H-=l[i]*i/(max-1)*math.log(i/(max-1),2)
    return round(H,2)

with open('test_data.csv') as file:
    data = file.read()

print("Data in file (test):")
print(data)

print("Out:")
print(task(data))