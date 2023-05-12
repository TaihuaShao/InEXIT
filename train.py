import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from dataset import load_sents, PJFDataset
from model import BertMatchingModel
from transformers import AutoModel, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

def main():
    args = parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    elif os.path.exists(os.path.join(args.log_dir, '{}-seed{}.log.data'.format(args.token, str(args.seed)))):
        os.remove(os.path.join(args.log_dir, '{}-seed{}.log.data'.format(args.token, str(args.seed))))
        
    logger = get_logger(os.path.join(args.log_dir, '{}-seed{}.log.data'.format(args.token, str(args.seed))))
    logger.info(args)
    
    def set_seed(seed):
        torch.manual_seed(seed) # cpu
        torch.cuda.manual_seed_all(seed)  # gpu
        torch.backends.cudnn.deterministic = True

    logger.info('PYTORCH VERSION: {}.'.format(torch.__version__))    
    logger.info('Starting training [{}].'.format(args.token))
    logger.info("Seed for training: " + str(args.seed))
    
    set_seed(args.seed)

    geek_sents = load_sents('geek', args)
    job_sents = load_sents('job', args)
    
    train_dataset = PJFDataset(geek_sents, job_sents, args, token='train')
    valid_dataset = PJFDataset(geek_sents, job_sents, args, token='valid')
    test_dataset = PJFDataset(geek_sents, job_sents, args, token='test')

    logger.info("Number of avaliable training samples: " + str(len(train_dataset)))
    logger.info("Number of avaliable validing samples: " + str(len(valid_dataset)))
    logger.info("Number of avaliable testing samples: " + str(len(test_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        pin_memory =True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.valid_num_workers,
        pin_memory =True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        pin_memory =True
    )

    logger.info('Finishing data preparation.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu)
    logger.info("Training on the device: " + str(device))
    
    model = BertMatchingModel(args).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logger.info(model)
    logger.info(get_parameter_number(model))

    logger.info('Starting train process.')

    best_acc = 0
    best_epoch = 0
    best_result = (0, 0, 0, 0, 0, 0)
    def metrics(text, label, pre, running_loss):
        roc_auc = roc_auc_score(label, pre)
        TP, FN, FP, TN = classify(label, pre)
        tot = TP + FN + FP + TN
        acc = (TP + TN) / tot
        if (TP + FN) != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        if (TP + FP) != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        if (precision + recall) != 0:
            f1score = 2 * precision * recall / (precision + recall)
        else:
            f1score = 0
        if text=="Test: ":
            print(str(TP) + '\t' + str(FN) + '\t' + str(FP) + '\t' + str(TN))
            epoch_info = text+'[epoch-{}]\tTesting Loss:\t{}\n\tROC_AUC:\t{}\n\tACC:\t\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}'.format(
                best_epoch, running_loss, roc_auc, acc, precision, recall, f1score)
        elif text =="Train: ":
            print(str(TP) + '\t' + str(FN) + '\t' + str(FP) + '\t' + str(TN))
            epoch_info = text+'Epoch [{}/{}]\tTraining Loss:\t{}\n\tROC_AUC:\t{}\n\tACC:\t\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}'.format(
                epoch+1, args.num_epochs, running_loss, roc_auc, acc, precision, recall, f1score)            
        else:
            print(str(TP) + '\t' + str(FN) + '\t' + str(FP) + '\t' + str(TN))
            epoch_info = text+'[epoch-{}]\tValiding Loss:\t{}\n\tROC_AUC:\t{}\n\tACC:\t\t{}\n\tPrecision:\t{}\n\tRecall:\t\t{}\n\tF1 Score:\t{}'.format(
                epoch+1, running_loss, roc_auc, acc, precision, recall, f1score)
        logger.info(epoch_info)
        sys.stdout.flush()
        return acc, roc_auc, precision, recall, f1score

    total_step = len(train_loader)
    logger.info("Total_step in one epoch: " + str(total_step))

    start_epoch = -1

    for epoch in range(start_epoch + 1, args.num_epochs):
        running_loss = 0.0
        model.train()
        pre_all = []
        label_all = []
        for i, (geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask, job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask, geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask, job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask, labels) in enumerate(train_loader):
            geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask, job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask, geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask, job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask, labels = geek_tokens_input_ids.to(device), geek_tokens_token_type_ids.to(device), geek_tokens_attention_mask.to(device), job_tokens_input_ids.to(device), job_tokens_token_type_ids.to(device), job_tokens_attention_mask.to(device), geek_sent_tokens_input_ids.to(device), geek_tokens_sent_token_type_ids.to(device), geek_sent_tokens_attention_mask.to(device), job_sent_tokens_input_ids.to(device), job_sent_tokens_token_type_ids.to(device), job_sent_tokens_attention_mask.to(device), labels.to(device)
            outputs = model(geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask, job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask, geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask, job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask)
            loss = criterion(outputs, labels)

            pre = [x.item() for x in outputs.cpu()]
            label = [x.item() for x in labels.cpu()]
            pre_all += pre
            label_all += label

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)            
            optimizer.step()
            running_loss += loss.item()
            if (i+1)%250 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format( epoch+1, args.num_epochs, i+1, total_step, running_loss/(i+1) ))
                sys.stdout.flush()
        metrics("Train: ", label_all, pre_all, running_loss/len(train_loader) )        
        sys.stdout.flush()

        pre_all = []
        label_all = []

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for (geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask, job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask, geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask, job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask, labels) in valid_loader:
                geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask, job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask, geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask, job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask, labels = geek_tokens_input_ids.to(device), geek_tokens_token_type_ids.to(device), geek_tokens_attention_mask.to(device), job_tokens_input_ids.to(device), job_tokens_token_type_ids.to(device), job_tokens_attention_mask.to(device),  geek_sent_tokens_input_ids.to(device), geek_tokens_sent_token_type_ids.to(device), geek_sent_tokens_attention_mask.to(device), job_sent_tokens_input_ids.to(device), job_sent_tokens_token_type_ids.to(device), job_sent_tokens_attention_mask.to(device), labels.to(device)
                outputs = model(geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask, job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask, geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask, job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask)
                loss = criterion(outputs, labels)
                outputs = torch.sigmoid(outputs)
                pre = [x.item() for x in outputs.cpu()]
                label = [x.item() for x in labels.cpu()]
                pre_all += pre
                label_all += label
                valid_loss += loss.item()

        acc, roc_auc, precision, recall, f1score = metrics("Eval ", label_all, pre_all, valid_loss/len(valid_loader))

        checkpoint = {
        "net": model.state_dict(),
        'optimizer':optimizer.state_dict(),
        "epoch": epoch
        }

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_result = (best_epoch, roc_auc, acc, precision, recall, f1score)
        
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)  
        torch.save(checkpoint, os.path.join(args.save_path, '{}-seed{}-model-{}.pth.tar'.format(args.token, str(args.seed), int(epoch+1))))

        if epoch+1 >= best_epoch + args.end_step or epoch+1 >= args.num_epochs:
            logger.info('Finish after epoch {}'.format(epoch+1))
            sys.stdout.flush()
            keep_only_the_best(args, best_epoch)
            out_epoch, roc_auc, acc, precision, recall, f1score = best_result
            best_epoch_info = 'BEST Eval: [epoch-{}]\n\tROC_AUC:\t{:.4f}\n\tACC:\t\t{:.4f}\n\tPrecision:\t{:.4f}\n\tRecall:\t\t{:.4f}\n\tF1 Score:\t{:.4f}'.format(
                out_epoch, roc_auc, acc, precision, recall, f1score)
            logger.info(best_epoch_info)
            break

    # test
    model.load_state_dict(torch.load(os.path.join(args.save_path, '{}-seed{}-model-best.pth.tar'.format(args.token, str(args.seed))))['net'])
    model.eval()

    pre_all = []
    label_all = []
    
    test_loss = 0.0
    with torch.no_grad():
        for (geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask, job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask, geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask, job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask, labels) in test_loader:
            geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask, job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask, geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask, job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask, labels = geek_tokens_input_ids.to(device), geek_tokens_token_type_ids.to(device), geek_tokens_attention_mask.to(device), job_tokens_input_ids.to(device), job_tokens_token_type_ids.to(device), job_tokens_attention_mask.to(device), geek_sent_tokens_input_ids.to(device), geek_tokens_sent_token_type_ids.to(device), geek_sent_tokens_attention_mask.to(device), job_sent_tokens_input_ids.to(device), job_sent_tokens_token_type_ids.to(device), job_sent_tokens_attention_mask.to(device), labels.to(device)
            outputs = model(geek_tokens_input_ids, geek_tokens_token_type_ids, geek_tokens_attention_mask, job_tokens_input_ids, job_tokens_token_type_ids, job_tokens_attention_mask, geek_sent_tokens_input_ids, geek_tokens_sent_token_type_ids, geek_sent_tokens_attention_mask, job_sent_tokens_input_ids, job_sent_tokens_token_type_ids, job_sent_tokens_attention_mask)
            loss = criterion(outputs, labels)
            outputs = torch.sigmoid(outputs)
            pre = [x.item() for x in outputs.cpu()]
            label = [x.item() for x in labels.cpu()]
            pre_all += pre
            label_all += label
            test_loss += loss.item()

        with open('seed{}.pre.labels'.format(str(args.seed)), 'w', encoding='utf-8') as file:
            for i in range(0,len(pre_all)):
                file.write(str(label_all[i]) + '\t' + str(pre_all[i]) + '\n')

    metrics("Test: ", label_all, pre_all, test_loss/len(test_loader))
        
if __name__ == '__main__':
    main()
