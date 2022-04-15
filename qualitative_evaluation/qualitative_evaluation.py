import json
import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt


# Reading in the clinical evalution resutls:
with open('evaluation_1.json') as json_file:
    data1 = json.load(json_file)
with open('evaluation_2.json') as json_file:
    data2 = json.load(json_file)
with open('evaluation_3.json') as json_file:
    data3 = json.load(json_file)
with open('evaluation_4.json') as json_file:
    data4 = json.load(json_file)
with open('evaluation_5.json') as json_file:
    data5 = json.load(json_file)
with open('evaluation_6.json') as json_file:
    data6 = json.load(json_file)
with open('evaluation_7.json') as json_file:
    data7 = json.load(json_file)
with open('evaluation_8.json') as json_file:
    data8 = json.load(json_file)
with open('evaluation_9.json') as json_file:
    data9 = json.load(json_file)

datas = [data1, data4, data6, data8, data9, data5, data2, data7, data3]

# Reding the majority vote results of our algorithm
with open('majority_vote.json') as json_file:
    maj_votes = json.load(json_file)

diagnosis_total = {'overall': 0 ,'akiec': 0, 'bcc': 0, 'bkl': 0, 'df': 0, 'mel': 0, 'nv': 0, 'vasc': 0}
accuracy_task1_total = {'overall': 0 ,'akiec': 0, 'bcc': 0, 'bkl': 0, 'df': 0, 'mel': 0, 'nv': 0, 'vasc': 0}
accuracy_task2_total = {'overall': 0 ,'akiec': 0, 'bcc': 0, 'bkl': 0, 'df': 0, 'mel': 0, 'nv': 0, 'vasc': 0}
majority_vote_total = {'overall': 0 ,'akiec': 0, 'bcc': 0, 'bkl': 0, 'df': 0, 'mel': 0, 'nv': 0, 'vasc': 0}

diagnoses_listed = []
task1_listed = []
task2_listed = []

confidence_task1_total = {'overall': [], 'correct': [], 'wrong': []}
confidence_task2_total = {'overall': [], 'correct': [], 'wrong': []}

PCPS = {'PCP1': None, 'PCP2': None, 'PCP3': None, 'PCP4': None, 'PCP5': None, 'PCP6': None, 'PCP7': None, 'PCP8': None, 'PCP9': None, 'avg.': None}
qualitative_results = {'Majority-vote': {'accuracy': PCPS.copy()}, 'Task 1': {'accuracy': PCPS.copy(), 'confidence': PCPS.copy()}, 'Task 2': { 'accuracy': PCPS.copy(), 'confidence': PCPS.copy() }}

for (j, data) in enumerate(datas):
    majority_vote = {'overall': 0 ,'akiec': 0, 'bcc': 0, 'bkl': 0, 'df': 0, 'mel': 0, 'nv': 0, 'vasc': 0}
    accuracy_task1 = {'overall': 0 ,'akiec': 0, 'bcc': 0, 'bkl': 0, 'df': 0, 'mel': 0, 'nv': 0, 'vasc': 0}
    accuracy_task2 = {'overall': 0 ,'akiec': 0, 'bcc': 0, 'bkl': 0, 'df': 0, 'mel': 0, 'nv': 0, 'vasc': 0}
    diagnosis = {'overall': len(data['Diagnosis']), 'akiec': data['Diagnosis'].count('akiec'), 'bcc': data['Diagnosis'].count('bcc'),
                'bkl': data['Diagnosis'].count('bkl'), 'df': data['Diagnosis'].count('df'), 'mel': data['Diagnosis'].count('mel'),
                'nv': data['Diagnosis'].count('nv'), 'vasc': data['Diagnosis'].count('vasc')}
    confidence_task1 = {'overall': data['ImConf'], 'correct': [], 'wrong': []}
    confidence_task2 = {'overall': data['RetConf'], 'correct': [], 'wrong': []}


    for (i, diag) in enumerate(data['Diagnosis']):

        maj_vote = maj_votes[data['ImgName'][i][4:-4] + '.png']
        # check task 1
        if diag == data['Image'][i]:
            accuracy_task1['overall'] += 1
            accuracy_task1[diag] += 1
            confidence_task1['correct'].append(confidence_task1['overall'][i])
        else: 
            confidence_task1['wrong'].append(confidence_task1['overall'][i])
        # check task 2
        if diag == data['Retrieval'][i]:
            accuracy_task2['overall'] += 1
            accuracy_task2[diag] += 1
            confidence_task2['correct'].append(confidence_task2['overall'][i])
        else: 
            confidence_task2['wrong'].append(confidence_task2['overall'][i])
        # check majority vote
        if diag == maj_vote:
            majority_vote['overall'] += 1
            majority_vote[diag] += 1

    for key in diagnosis_total:
        diagnosis_total[key] += diagnosis[key]
        accuracy_task2_total[key] += accuracy_task2[key]
        accuracy_task1_total[key] += accuracy_task1[key]
        majority_vote_total[key] += majority_vote[key]
    for key in confidence_task1_total:
        confidence_task1_total[key] += confidence_task1[key]
        confidence_task2_total[key] += confidence_task2[key]

    diagnoses_listed += data['Diagnosis']
    task1_listed += data['Image']
    task2_listed += data['Retrieval']

    #write results to dictionary
    qualitative_results['Majority-vote']['accuracy']['PCP'+str(j+1)] = majority_vote['overall']
    qualitative_results['Task 1']['accuracy']['PCP'+str(j+1)] = accuracy_task1['overall']
    qualitative_results['Task 2']['accuracy']['PCP'+str(j+1)] = accuracy_task2['overall']
    qualitative_results['Task 1']['confidence']['PCP'+str(j+1)] = {}
    qualitative_results['Task 1']['confidence']['PCP'+str(j+1)]['overall'] = sum(confidence_task1['overall'])/len(confidence_task1['overall'])
    qualitative_results['Task 1']['confidence']['PCP'+str(j+1)]['correct'] = sum(confidence_task1['correct'])/len(confidence_task1['correct'])
    qualitative_results['Task 1']['confidence']['PCP'+str(j+1)]['wrong'] = sum(confidence_task1['wrong'])/len(confidence_task1['wrong'])
    qualitative_results['Task 2']['confidence']['PCP'+str(j+1)] = {}
    qualitative_results['Task 2']['confidence']['PCP'+str(j+1)]['overall'] = sum(confidence_task2['overall'])/len(confidence_task2['overall'])
    qualitative_results['Task 2']['confidence']['PCP'+str(j+1)]['correct'] = sum(confidence_task2['correct'])/len(confidence_task2['correct'])
    qualitative_results['Task 2']['confidence']['PCP'+str(j+1)]['wrong'] = sum(confidence_task2['wrong'])/len(confidence_task2['wrong'])

    # Print resutls of Participant j
    print('-------------------- Participant ', j+1, ' -------------------')       
    print('Accuracies:   Majority vote = ', majority_vote['overall'], '%  Task 1 = ', accuracy_task1['overall'], '%  Task 2 = ', accuracy_task2['overall'], '%' )
    print('Confidence Task 1 (overall (correct/wrong)): ',  f"{sum(confidence_task1['overall'])/len(confidence_task1['overall']) :.2f}", ' ( ', 
                                                            f"{sum(confidence_task1['correct'])/len(confidence_task1['correct']) :.2f}",' / ', 
                                                            f"{sum(confidence_task1['wrong'])/len(confidence_task1['wrong']) :.2f}",' )'  )
    print('Confidence Task 2 (overall (correct/wrong)): ',  f"{sum(confidence_task2['overall'])/len(confidence_task2['overall']) :.2f}", ' ( ', 
                                                            f"{sum(confidence_task2['correct'])/len(confidence_task2['correct']) :.2f}",' / ', 
                                                            f"{sum(confidence_task2['wrong'])/len(confidence_task2['wrong']) :.2f}",' )'  )


print('---------------------- Overall  ---------------------')       
print('Accuracies:   Majority vote = ', f"{majority_vote_total['overall']/9.0 :.1f}", '%  Task 1 = ', f"{accuracy_task1_total['overall']/9.0 :.1f}", '%  Task 2 = ', f"{accuracy_task2_total['overall']/9.0 :.1f}", '%' )
print('Confidence Task 1 (overall (correct/wrong)): ',  f"{sum(confidence_task1_total['overall'])/len(confidence_task1_total['overall']) :.2f}", ' ( ', 
                                                        f"{sum(confidence_task1_total['correct'])/len(confidence_task1_total['correct']) :.2f}",' / ', 
                                                        f"{sum(confidence_task1_total['wrong'])/len(confidence_task1_total['wrong']) :.2f}",' )'  )
print('Confidence Task 2 (overall (correct/wrong)): ',  f"{sum(confidence_task2_total['overall'])/len(confidence_task2_total['overall']) :.2f}", ' ( ', 
                                                        f"{sum(confidence_task2_total['correct'])/len(confidence_task2_total['correct']) :.2f}",' / ', 
                                                        f"{sum(confidence_task2_total['wrong'])/len(confidence_task2_total['wrong']) :.2f}",' )'  )


qualitative_results['Majority-vote']['accuracy']['avg.'] = majority_vote_total['overall']/9.0
qualitative_results['Task 1']['accuracy']['avg.'] = accuracy_task1_total['overall']/9.0
qualitative_results['Task 2']['accuracy']['avg.'] = accuracy_task2_total['overall']/9.0
qualitative_results['Task 1']['confidence']['avg.'] = {}
qualitative_results['Task 2']['confidence']['avg.'] = {}
qualitative_results['Task 1']['confidence']['avg.']['overall'] = sum(confidence_task1_total['overall'])/len(confidence_task1['overall'])
qualitative_results['Task 1']['confidence']['avg.']['correct'] = sum(confidence_task1_total['correct'])/len(confidence_task1['correct'])
qualitative_results['Task 1']['confidence']['avg.']['wrong'] = sum(confidence_task1_total['wrong'])/len(confidence_task1_total['wrong'])
qualitative_results['Task 2']['confidence']['avg.']['overall'] = sum(confidence_task2_total['overall'])/len(confidence_task2_total['overall'])
qualitative_results['Task 2']['confidence']['avg.']['correct'] = sum(confidence_task2_total['correct'])/len(confidence_task2_total['correct'])
qualitative_results['Task 2']['confidence']['avg.']['wrong'] = sum(confidence_task2_total['wrong'])/len(confidence_task2_total['wrong'])

with open('../results/qualitative_results.json', 'w') as fp:
    json.dump(qualitative_results, fp, indent=2)

# Plotting the confusion matrices

cm = confusion_matrix(diagnoses_listed, task1_listed, normalize='true')
cm2 = confusion_matrix(diagnoses_listed, task2_listed, normalize='true')

# just for annotation:
cm_no = confusion_matrix(diagnoses_listed, task1_listed)
cm2_no = confusion_matrix(diagnoses_listed, task2_listed)
df_cm_no = pd.DataFrame(cm_no, ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])
df_cm_no.mask(df_cm_no < 0.005, inplace=True)
df_cm2_no = pd.DataFrame(cm2_no, ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])
df_cm2_no.mask(df_cm2_no < 0.005, inplace=True)

df_cm = pd.DataFrame(cm, ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])
df_cm.mask(df_cm < 0.005, inplace=True)
df_cm2 = pd.DataFrame(cm2, ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])
df_cm2.mask(df_cm2 < 0.005, inplace=True)


sn.set(font_scale=1.4) # for label size
sn.set(rc={'axes.facecolor':'white'})

ax = sn.heatmap(df_cm, annot=df_cm_no, annot_kws={"size": 12}, fmt='.0f', robust=True, cmap="YlGnBu", vmin=0, vmax=1) # font size
cbar = ax.collections[0].colorbar
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
plt.savefig('../results/cm_task1.png')
plt.clf()


ax2 = sn.heatmap(df_cm2, annot=df_cm2_no, annot_kws={"size": 12}, fmt='.0f', robust=True, cmap="YlGnBu", vmin=0, vmax=1) # font size
cbar2 = ax2.collections[0].colorbar
cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1])
cbar2.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
plt.savefig('../results/cm_task2.png')
plt.clf()
