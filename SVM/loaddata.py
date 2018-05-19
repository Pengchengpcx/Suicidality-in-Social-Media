'''
Prepare training data for the SVM classifier
Combine all posts of the same user together
'''
import csv
import pickle

testfiles = './testingdata/testing.csv'
trainfiles = './trainingdata/trainingless.csv'

def create_data(path, savepath):
    with open(path) as f:
        users = []
        user = []
        labels = [0]
        cur_user = -20001

        reader = csv.DictReader(f)

        i = 1

        for row in reader:
            if row['user_id'] == 'user_id':
                continue
            print((row['user_id']))

            if int(row['user_id']) == cur_user:
                user.append(row['post_body'])
            else:
                users.append(user)
                user = []
                cur_user = int(row['user_id'])
                user.append(row['post_body'])
                labels.append(int(row['label']))

                i += 1

        users.append(user)

    posts = []

    for user in users:
        posts.append(' '.join(user))

    print(len(users), len(posts), len(labels), i)

    pickle.dump((posts, labels), open(savepath, 'wb'))

create_data(trainfiles, './svm_train')



