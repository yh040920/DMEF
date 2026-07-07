import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--AD_dir', type=str,
                        help='subfloder of train or test dataset', default='AD/')
    parser.add_argument('--CN_dir', type=str,
                        help='subfloder of train or test dataset', default='CN/')
    parser.add_argument('--MCI_dir', type=str,
                        help='subfloder of train or test dataset', default='MCI/')
    parser.add_argument('--PMCI_dir', type=str,
                        help='subfloder of train or test dataset', default='PMCI/')
    parser.add_argument('--SMCI_dir', type=str,
                        help='subfloder of train or test dataset', default='SMCI/')

    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.001)')
    parser.add_argument('--class_num', type=int, help='class_num', default=2)
    parser.add_argument('--seed', type=int, help='Seed', default=42)
    parser.add_argument('--gpu', type=str, help='GPU ID', default='0')
    parser.add_argument('--device', type=str, help='model_name', default='cuda:0')
    parser.add_argument('--train_root_path', type=str, help='Root path for train dataset',
                        default=r'/3251903008/yyh/image1/train')
    parser.add_argument('--val_root_path', type=str, help='Root path for val dataset',
                        default=r'/3251903008/yyh/image1/val')
    parser.add_argument('--test_root_path', type=str, help='Root path for test dataset',
                        default=r'/3251903008/yyh/image1/test')
    parser.add_argument('--excel_file', type=str, help='excel file for scores',
                        default=r'/3251903008/yyh/ADNIbase1416_info.xlsx')
    parser.add_argument('--batch_size', type=int, help='batch_size of data', default=8)
    parser.add_argument('--nepoch', type=int, help='Total epoch num', default=90)
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading', default=4)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.3)
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation', default=True)
    # parser.add_argument('--model_path', type=str, help='Path to the pretrained model file (.pt)', default="/3251903008/yyh/result/buchong_20251227_073806/models/3DNet4l_1_tp4_sifeature_clonefs_1bt1_l43_1_epoch_60_d1_det.pt")


    return parser.parse_args()
