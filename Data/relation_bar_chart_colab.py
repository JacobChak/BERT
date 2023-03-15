# -*- coding: utf-8 -*-
# Draw a statistical bar chart of the frequency of character relationships
import pandas as pd
import matplotlib.pyplot as plt

# Insert the data from EXCEL
df = pd.read_excel('D:\LegalAI\data\人物关系表.xlsx')
label_list = list(df['rel'].value_counts().index)
num_list = df['rel'].value_counts().tolist()

# Use the Matplotlib module to draw a bar chart
x = range(len(num_list))
rects = plt.bar(x=x, height=num_list, width=0.6, color='blue', label="Frequency")

plt.ylabel("Amount")
plt.xticks([index + 0.1 for index in x], label_list)
plt.xticks(rotation=45)    
plt.xlabel("Relationship")
plt.title("Relationship Frequency Statistics")
plt.legend()

# Text description of the bar chart
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

# plt.show()
plt.savefig('D:/LegalAI/data/bar_chart.png')
