import matplotlib.pyplot as plt
import glob
import numpy as np


result_files = glob.glob('results/*')

all_train_errs = {10: [], 20: [], 30: [], 40: [], 50: [], 100: [], 200: [], 300: [], 400: [], 500: [], 1000: [], 2000: [], 3000: [], 4000: [], 5000: []}
all_test_errs = {10: [], 20: [], 30: [], 40: [], 50: [], 100: [], 200: [], 300: [], 400: [], 500: [], 1000: [], 2000: [], 3000: [], 4000: [], 5000: []}
k_index = [10,20,30,40,50,100,200,300,400,500,1000,2000,3000,4000,5000]

best_test_err = 0
best_file = ''

for resultf in result_files:
    with open(resultf, 'r') as results:
        lines = results.readlines()
        if len(lines) < 2:
            continue
        k = int(lines[2][3:])
        train_err = 1-float(lines[3].split(' ')[9])
        test_err = 1-float(lines[3].split(' ')[-1])

        all_train_errs[k].append(train_err)
        all_test_errs[k].append(test_err)

        if test_err > best_test_err:
            best_test_acc = test_err       
            best_file = resultf

train_means = [0 for i in range(len(k_index))]
test_means = [0 for i in range(len(k_index))]
train_std = [0 for i in range(len(k_index))]
test_std = [0 for i in range(len(k_index))]

for k, values in all_train_errs.items():
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)

    index = k_index.index(k)
    train_means[index] = mean
    train_std[index] = std

for k, values in all_test_errs.items():
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)

    index = k_index.index(k)
    test_means[index] = mean
    test_std[index] = std

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
train_line = ax.plot(k_index, train_means, label='Train', color='b')
test_line = ax.plot(k_index, test_means, label='Test', color='g')
baseline_error = 0.48
baseline = ax.plot(k_index, [baseline_error for i in range(len(k_index))], label='Baseline', color='r')
# ax.legend([train_line, test_line, baseline], labels=['Train', 'Test', 'Baseline'])

ax.errorbar(k_index, train_means, yerr=train_std)
ax.errorbar(k_index, test_means, yerr=test_std)
ax.set_xlabel('Num Clusters')
ax.set_ylabel('Error')
ax.set_title('Error vs Num Clusters')
ax.set_xscale('log')
plt.legend()

plt.show()

print best_file










