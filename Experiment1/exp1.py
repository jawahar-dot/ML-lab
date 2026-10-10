import csv
import os

a = []
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'enjoysport.csv')

with open(csv_path, 'r') as csvfile:
    for row in csv.reader(csvfile):
        print(a)
        a.append(row)
print("\n The total number of training instances are: ",len(a))

num_attribute = len(a[0])-1
print("\n The initial hypothesis is: ") 
hypothesis = ['0'] * num_attribute
print(hypothesis)

for i in range(0, len(a)):
    if a[i][num_attribute] == 'yes':
        for j in range(0, num_attribute):
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'

print("\n The hypothesis for the training instance {} is: \n".format(i+1), hypothesis)
print("\n The Maximally specific hypothesis for the training instance is ")
print(hypothesis)
