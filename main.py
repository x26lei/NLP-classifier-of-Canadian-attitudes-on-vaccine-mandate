from loaddata import *
from random_batch import *
from dnn import *
x_t, y_t, x_test, y_test, x_whole, y_whole= load_data('Yes_Sep.xlsx', 'No_Sep.xlsx', 500)
# x_t 是一个列表，里面每一个元素都是一个tensor， y_t 是一个列表，里面每一个元素都是对应x_t中张量的标签
x_train_test, y_train_test = random_batch(x_t, y_t, len(x_t))
batch, batch_label = random_batch(x_t, y_t, 5)
test_data, test_label = random_batch(x_test, y_test, len(x_test))
test_whole, label_whole = random_batch(x_whole, y_whole, len(x_whole))
loss_avg = []
acc_avg = []
acc1_avg =[]
num_epoch = 80
for i in range (0, 1):
    net = net_init()
    loss_list = []
    acc_list = []
    acc1_list = []
    for epoch in range(0, num_epoch):
        loss = train(net, batch, batch_label)
        # acc = test_model(net, test_data, test_label)
        acc = test_model(net, x_train_test, y_train_test)
        acc1 = test_model(net, test_whole, label_whole)
        print('epoch: {}, loss: {}, test accuracy: {}, test accuracy whole set: {}'.format(epoch, loss, acc, acc1))
        loss_list.append(loss)
        acc_list.append(acc)
        acc1_list.append(acc1)
        torch.save(net, 'net500.pkl')
    if i == 0:
        loss_avg = loss_list
        acc_avg = acc_list
        acc1_avg = acc1_list
    else:
        #loss_avg = np.sum([loss_avg, loss_list],axis=0).tolist()
        #acc_avg = np.sum([acc_avg, acc_list], axis=0).tolist()
        #acc1_avg = np.sum([acc1_avg, acc1_list], axis=0).tolist()
        loss_avg = np.sum([loss_avg, loss_list], axis=0)
        acc_avg = np.sum([acc_avg, acc_list], axis=0)
        acc1_avg = np.sum([acc1_avg, acc1_list], axis=0)
#loss_avg = loss_avg / 3
#acc_avg = acc_avg / 3
#acc1_avg = acc1_avg / 3
plt.plot(np.arange(0, num_epoch), loss_avg, 'r-x', label='Loss')
plt.plot(np.arange(0, num_epoch), acc_avg, 'g-^', label='Accuracy of training dataset')
plt.plot(np.arange(0, num_epoch), acc1_avg, 'b-o', label='Accuracy of entire dataset')
plt.legend()
plt.xlabel('Epoch')
plt.show()
