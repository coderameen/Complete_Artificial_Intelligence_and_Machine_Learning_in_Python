#pip install matplotlib
#"it is used to Analylize and visualization of data/dataset/dataframe"


import matplotlib.pyplot as plt

#Bar Graph
#Real world usecase of student marks visualization
#x => students and y=>marks

x = [1,2,3,4]
y =[58,78,94,12]
label = ['Second Class', 'First Class','Distinction','Fail']


#bar plot
# plt.bar(x,y, tick_label=label,width=0.5, color=['purple','orange','green','red'])
# plt.xlabel('Students')
# plt.ylabel('Marks')
# plt.title("Student marks visualization")
# # plt.show()

'''
#Pie Chart
contents = ['Plants','Animal','Water']
colors = ['green','brown','blue']
slices = [3,7,8]
plt.pie(slices,labels=contents,colors=colors, startangle=90,autopct="%1.3f%%", explode=(0,0,0.05), shadow=True)
plt.legend()
plt.show()
'''




#Histogram
'''
percentage = [25,5,45,67,86,34,23,45,78,44,11]
range = (0,100)
num = 5
plt.hist(percentage,num,range,color='orange',histtype='bar')
plt.show()
'''

#scatter plot
x = [1,2,3,4,5,6,7]
y = [3,4,5,6,8,9,10]
plt.scatter(x,y,color='orange',marker='*')
plt.xlabel("x-axis")
plt.ylabel('y-axis')
plt.title('my scatter plot')
plt.show()