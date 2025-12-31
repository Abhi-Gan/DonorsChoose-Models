import pandas as pd

projects_data = pd.read_csv("dataset/train.csv")
# find the number of projects approved out of total number of projects
approved = 0
total = 0
"""for approvalStatus in projects_data['project_is_approved']:
    if approvalStatus == 1:
        approved += 1
    total += 1
print("Approved: %d" % approved)
print("Total: %d" % total)"""
numMen = 0
approvedMen = 0
numWomen = 0
approvedWomen = 0
numOther = 0
approvedOther = 0
for project in projects_data.iloc:
    if project['teacher_prefix'] in ["Ms.", "Mrs."]:
        numWomen += 1
        if project['project_is_approved'] == 1:
            approvedWomen += 1
    elif project['teacher_prefix'] in ["Mr."]:
        numMen += 1
        if project['project_is_approved'] == 1:
            approvedMen += 1
    else:
        numOther += 1
        if project['project_is_approved'] == 1:
            approvedOther += 1

print("Numbers:\n Women - %d\n Men - %d\n Unknown - %d" % (numWomen, numMen, numOther))
print("Approval Proportions:\n Women - %f\n Men - %f\n Unknown - %f\n Total - %f"% (approvedWomen/numWomen, approvedMen/numMen, approvedOther/numOther, (approvedMen+approvedWomen+approvedOther)/(numMen+numWomen+numOther)))
