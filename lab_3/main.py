from io import StringIO
import csv

def task(csv_string):
    f = StringIO(csv_string)
    reader = csv.reader(f, delimiter=',')
    out = []
    for row in reader:
        out.append(row)
    r1=[]
    r2=[]
    for item in out:
        item[0]=int(item[0])
        item[1]=int(item[1])
        if item[0] not in r1:
            r1.append(item[0])
        if item[1] not in r2:
            r2.append(item[1])
    r3=[]
    r4=[]
    for item in out:
        for it in out:
            if item[1]==it[0]:
             if item[0] not in r3:
                r3.append(item[0])
             if it[1] not in r4:
                r4.append(it[1])
    r5=[]
    for item in out:
        for it in out:
            if item[0]==it[0] and item!=it:
                if item[1] not in r5:
                    r5.append(item[1])
                if it[1] not in r5:
                    r5.append(it[1])
                a=item[1]
                b=it[1]
                while a in r1 and b in r1:
                    for elem in out:
                        if elem[0]==a:
                            if elem[1] not in r5:
                                r5.append(elem[1])
                                a1=elem[1]
                        if elem[0]==b:
                            if elem[1] not in r5:
                                r5.append(elem[1])
                                b1=elem[1]
                    a=a1
                    b=b1
    res=[r1,r2,r3,r4,r5]
    for mas in res:
        mas.sort()
    return res


with open('test_data.csv') as file:
    data = file.read()

print("Data in file (test):")
print(data)

print("Out:")
print(task(data))