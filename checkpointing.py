class Checkpoint(object):
    def __init__(self, model_name='', checkpoint_path_model=''):
        # self.best_acc = 0.0
        self.folder = 'checkpoint'
        self.model_name = model_name
        self.checkpoint_path = str(checkpoint_path_model)
        os.makedirs(self.folder, exist_ok=True)

    def save(self, model='', epoch=-1, res_train_loss=0, res_val_loss=0, train_loss=0, val_loss=0):
        # if acc > self.best_acc:
        print('Saving checkpoint...')
        state = {
            'net': model.state_dict(),
            'res_train_loss': res_train_loss,
            'res_val_loss': res_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch': epoch,
        }
        path = os.path.join(os.path.abspath(self.folder), self.model_name + '_' + str(epoch) +'.pth')
        torch.save(state, path)
        
        str_path = str(path)
        # print(str_path)
        !cp -av "$str_path" "$self.checkpoint_path"
        # self.best_acc = acc
        !rm -rf "$str_path"


    def load(self, model='', epoch=-1):
        drive_PATH = os.path.join(self.checkpoint_path, self.model_name + '_' + str(epoch) +'.pth')
        str_drive_PATH = str(drive_PATH)
        str_abs_PATH = str(os.path.abspath(self.folder))
        print(str_drive_PATH)
        print(str_abs_PATH)
        !cp -av "$str_drive_PATH" '/content/checkpoint'

        PATH = os.path.join(os.path.abspath(self.folder), self.model_name + '_' + str(epoch) +'.pth')
        str_path = str(PATH)
        
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['net'])
        
        
        
class LossHistory(object):
    def __init__(self, model_name='U_Net', checkpoint_path_loss=''):
        self.folder = 'Loss_History'
        self.model_name = model_name
        os.makedirs(self.folder, exist_ok=True)
        self.checkpoint_path = str(checkpoint_path_loss)
        self.file_name = None
        self.file_path = None

    def save(self, loss_history={}, start_epoch=0, num_epochs=100, epoch=-1):
        history_track = {}  # loss history
        history_track['train_loss'] = loss_history['train_loss']
        history_track['valid_loss'] = loss_history['valid_loss']
        history_track['res_train_loss'] = loss_history['res_train_loss']
        history_track['res_valid_loss'] = loss_history['res_valid_loss']
        history_track['start_epoch'] = start_epoch
        history_track['num_epochs'] = num_epochs
        history_track['epoch'] = epoch
        self.file_name = self.model_name + '_' + 'Loss' + '_' + str(epoch) + '.json'
        jason_file = open(os.path.join(os.path.abspath(self.folder), self.file_name), "w")
        json.dump(history_track, jason_file)
        jason_file.close()

        self.file_path = os.path.join(os.path.abspath(self.folder), self.file_name)
        str_path = str(self.file_path)
        # print(str_path)
        !cp -av "$str_path" "$self.checkpoint_path"
        # self.best_acc = acc
        # !rm -rf "$str_path"
    
    def load(self, epoch=-1):
        self.file_name = self.model_name + '_' + 'Loss' + '_' + str(epoch) + '.json'
        drive_PATH = os.path.join(self.checkpoint_path, self.file_name)
        str_drive_PATH = str(drive_PATH)
        str_abs_PATH = str(os.path.abspath(self.folder))

        # print(str_drive_PATH)
        # print(str_abs_PATH)
        !cp -av "$str_drive_PATH" '$str_abs_PATH'

        with open(os.path.join(str_abs_PATH, str(self.file_name)), 'r') as json_file:
            json_data = json.load(json_file)
            print("\nTrain Loos:=> ", json_data['train_loss'])
            print("\nTest Loss:=> ", json_data['valid_loss'])
            print("\nStart Epoch:=> ", json_data['start_epoch'])
            print("\nEpoch:=> ", json_data['epoch'])
            loss_history = {'train_loss':json_data['train_loss'],
                            'valid_loss':json_data['valid_loss'],
                            'res_train_loss':json_data['res_valid_loss'],
                            'res_valid_loss':json_data['res_valid_loss'],
                            }
            start_epoch = json_data['start_epoch']
            num_epochs = json_data['num_epochs']
            epoch = json_data['epoch']
        return loss_history, start_epoch, num_epochs, epoch
