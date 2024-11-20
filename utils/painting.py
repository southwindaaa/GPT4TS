import os  
import matplotlib.pyplot as plt  
import numpy as np  
from .metrics import metric


#绘制test结果对比图
def draw_comparasion(args,trues,preds,ii):
    print(preds.shape,trues.shape)
    preds_sample = preds[-1]
    trues_sample = trues[-1]
    print('draw shape',preds_sample.shape,trues_sample.shape)
    if args.features == 'MS':
        preds_sample = preds_sample[:,-1]
        trues_sample = trues_sample[:,-1]

    # print(preds_sample,'\n',trues_sample)
    print('draw shape',preds_sample.shape,trues_sample.shape)
    mae_sample, mse_sample, rmse_sample, mape_sample, mspe_sample, smape_sample, nd_sample = metric(preds_sample, trues_sample)

    # 创建一个绘图
    plt.figure(figsize=(12, 6))


    # 绘制 preds 和 trues 的曲线
    plt.plot(preds_sample, label='Predictions', alpha=0.7)
    plt.plot(trues_sample, label='True Values', alpha=0.7)
    # print('Predictions vs True Values feature: '+ str(feat_ids[random_index,0]))
    plt.title('Predictions vs True Values feature: ')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend(title=f'MAE: {mae_sample:.4f}\nMSE: {mse_sample:.4f}\nMAPE: {mape_sample:.4f}')

    result_folder = './results/predict_images/'+args.data_path.split('.')[0]+'/'+args.features
    if not os.path.exists(result_folder):        os.makedirs(result_folder)
    # 保存图像
    print(result_folder+'/'+args.model+'_'+args.data_path.split('.')[0]+'_'+args.features + '_'+str(args.pred_len)+'_'+str(ii)+'.jpg')
    plt.savefig(result_folder+'/'+args.model+'_'+args.data_path.split('.')[0]+'_'+args.features + '_'+str(args.pred_len)+'_'+str(ii)+'.jpg')

    return mae_sample,mse_sample,mape_sample

#绘制收敛图
def draw_losses(args,losses_array,tag,ii):
    # 绘制loss图像
    plt.figure(figsize=(10, 5))
    plt.plot(losses_array, label=tag+' Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(tag+'ing Loss over Epochs')
    plt.legend()
    plt.grid(True)

    result_folder = './results/loss_images/' + args.data_path.split('.')[0]+'/'+args.features
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 保存图像
    print(result_folder + '/'+args.model+'_' + args.target + '_' + str(args.pred_len) +'_'+args.features+'_'+str(ii)+'_'+tag +'_loss.png')
    plt.savefig(result_folder + '/'+args.model+'_' + args.target + '_' + str(args.pred_len) +'_'+args.features+'_'+str(ii)+'_'+tag +'_loss.png', dpi=300, format='png')

# 保存预测结果
def store_results(args,trues,preds,ii):
    result_folder = './results/result_files/'+args.data_path.split('.')[0]+'/'+args.features

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # 获取数组的shape
    pred_shape = preds.shape
    true_shape = trues.shape
    
    preds_file_name = result_folder + '/'+args.model+'_' + args.target + '_' + str(args.pred_len) +'_'+args.features+'_'+str(ii)+'_preds.npy'
    trues_file_name = result_folder + '/'+args.model+'_' + args.target + '_' + str(args.pred_len) +'_'+args.features+'_'+str(ii)+'_trues.npy'
    shape_file = result_folder + '/'+args.model+'_' + args.target + '_' + str(args.pred_len) +'_'+args.features+'_'+str(ii)+'_shape.txt'

    # 将shape写入txt文件
    with open(shape_file, 'w') as f:
        f.write(f'preds shape: {pred_shape}\n')
        f.write(f'trues shape: {true_shape}\n')
    
    np.save(preds_file_name, preds)
    np.save(trues_file_name, trues)

def stroe_mean_std(args,mae_samples,mse_samples,mape_samples):
    mae_mean = np.mean(mae_samples)
    mae_std = np.std(mae_samples, ddof=1)

    mse_mean = np.mean(mse_samples)
    mse_std = np.std(mse_samples)

    mape_mean = np.mean(mape_samples)
    mape_std = np.std(mape_samples) 


    file_name = './results/numerical_results/' +args.data_path.split('.')[0]+'_'+args.model+'_' + args.target + '_' +args.features+'_num_results.txt'
    with open(file_name,'a') as f:
        f.write(f'seq_len: {args.seq_len}, pred_len: {args.pred_len}\nmae_mean: {mae_mean}\tmae_std: {mae_std}\tmse_mean: {mse_mean}\tmse_std: {mse_std}\tmape_mean: {mape_mean}\tmape_std: {mape_std}\n')

def store_str(args,str):
    if args.use_test2==1:
        test_method  = 'single_test'
    else:
        test_method = 'whole test'
    file_name = '/root/Load_LLM-experiments/Mine_2/GPT4TS/results/mae_mses/' +test_method+'_'+args.data_path.split('.')[0]+'_'+args.model+'_' + args.target + '_' +args.features+'.txt'
    with open(file_name,'a') as f:
        f.write(str+'\n')
