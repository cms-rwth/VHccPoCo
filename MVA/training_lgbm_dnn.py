import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import argparse
import glob
print(lgb.__version__)
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, log_loss
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.keras import backend as K
from keras import regularizers
from keras.layers import Dense, BatchNormalization, LeakyReLU
from keras.optimizers import RMSprop
from imblearn.over_sampling import SMOTE




# Function to load data
def load_data(dir_path, signal_names, background_names):
    signal_files = []
    for signal_name in signal_names:
        signal_files.extend(glob.glob(f"{dir_path}/{signal_name}/**/*.parquet", recursive=True))
    
    all_files = glob.glob(f"{dir_path}/**/*.parquet", recursive=True)

    background_files = []
    for background_name in background_names:
        background_files.extend([file for file in all_files if background_name in file])

    data_files = [file for file in all_files if "DATA" in file]

    if not signal_files:
        raise ValueError(f"No signal files found in directory {dir_path}/{signal_names}")
    if not background_files:
        raise ValueError(f"No background files found in directory {dir_path}")
    if not data_files:
        raise ValueError(f"No DATA files found in directory {dir_path}")

    signal_dfs = [pd.read_parquet(file) for file in signal_files]
    background_dfs = [pd.read_parquet(file) for file in background_files]
    data_dfs = [pd.read_parquet(file) for file in data_files]

    signal_df = pd.concat(signal_dfs, ignore_index=True)
    background_df = pd.concat(background_dfs, ignore_index=True)
    data_df = pd.concat(data_dfs, ignore_index=True)
    
    print("Signal DataFrame columns:", signal_df.columns)
    print("Background DataFrame columns:", background_df.columns)
    print("Data DataFrame columns:", data_df.columns)

    return signal_df, background_df, data_df

def get_background_abbreviation(background_names):
    abbreviations = []
    if any('DY' in name or 'dy' in name for name in background_names):
        abbreviations.append('DY')
    if any('TT' in name or 'tt' in name for name in background_names):
        abbreviations.append('TT')
    if any('QCD' in name or 'qcd' in name for name in background_names):
        abbreviations.append('QCD')
    if not abbreviations:  # If no specific backgrounds were found, consider it as 'Other Backgrounds'
        return 'OB'
    return '_'.join(abbreviations)

def train_model(X, y, X_data, signal_name, background_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)
    valid_sets = [valid_data]

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': 2,
    }

    bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=valid_sets, 
                    callbacks=[lgb.early_stopping(stopping_rounds=10)])

    # Create directory for the model if it doesn't exist
    background_abbreviation = get_background_abbreviation(background_names)
    
    model_dir = f'Models/{signal_name}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model with the signal name
    model_path = os.path.join(model_dir, f'model_{background_abbreviation}.txt')
    bst.save_model(model_path)
    evaluate_model(bst, X_test, y_test, X_data, signal_name, 'lgbm')
    
    print("hello, you just made it here")

    return bst

def custom_binary_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(-y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred))

def train_dnn(X, y, X_data, signal_name, background_names):
    print(f"Original X: {X.shape[0]} rows")
    print(f"Original y: {y.shape[0]} rows")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(1, activation='sigmoid'))

    optimizer = RMSprop(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, callbacks=[early_stopping])
    
    background_abbreviation = get_background_abbreviation(background_names)
    
    model_dir = f'Models/{signal_name}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, f'dnn_model_{background_abbreviation}.h5')
    model.save(model_path)
    evaluate_model(model, X_test, y_test, X_data, signal_name, 'dnn')

    return model

# Function to evaluate model
def evaluate_model(model, X, y, X_data, signal_name, model_type):
    if model_type == 'lgbm':
        y_pred = model.predict(X)
        data_pred = model.predict(X_data)
    elif model_type == 'dnn':
        y_pred = model.predict(X).ravel()
        data_pred = model.predict(X_data).ravel()

    fpr, tpr, _ = roc_curve(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    loss = log_loss(y, y_pred)

    print(f'AUC: {auc}')
    print(f'Log Loss: {loss}')

    plot_dir = f'Models/{signal_name}/Plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure()
    plt.hist(y_pred[y == 0], bins=50, alpha=0.5, label='Background', density=True)
    plt.hist(y_pred[y == 1], bins=50, alpha=0.5, label='Signal', density=True)
    plt.hist(data_pred, bins=50, alpha=0.75, label='Data', density=True)
    plt.xlabel('Score')
    plt.legend()
    plt.savefig(f'{plot_dir}/Score_{model_type}.png')

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(f'{plot_dir}/roc_curve_{model_type}.png')


# Main function to load data and train/evaluate model
def main():
    parser = argparse.ArgumentParser(description='Train or evaluate a LightGBM model.')
    parser.add_argument('mode', type=str, help='Mode: "train" or "eval".')
    parser.add_argument('dir_path', type=str, help='Path to the directory containing input files.')
    parser.add_argument('--signal', type=str, nargs='+', help='Name of the signal data file.')
    parser.add_argument('--background', type=str, nargs='+', help='Name of the bkg data files.')
    parser.add_argument('--model', type=str, help='Path to the model file.')
    parser.add_argument('--model_type', type=str, default='lgbm', choices=['lgbm', 'dnn'], help='The type of model to train.')
    parser.add_argument('--separate_trainings', action='store_true',
                    help='Enable separate trainings for dilep_pt < 150 and dilep_pt > 150 regions')
    args = parser.parse_args()

    signal_df, background_df, data_df = load_data(args.dir_path, args.signal, args.background)
    
    print(f"Signal DataFrame shape: {signal_df.shape}")
    signal_df = signal_df.dropna()
    background_df = background_df.dropna()
    data_df = data_df.dropna()
    
    # Before the training or evaluation process
    if isinstance(args.signal, list) and len(args.signal) > 1:
        # Concatenate signal names with a delimiter for readability
        concatenated_signals = "_".join(sorted(args.signal))  # Sort to ensure consistency
        common_signal_name = f"signalCombo_{concatenated_signals}"  # Prefix with 'signalCombo_' for clarity
    else:
        common_signal_name = args.signal if isinstance(args.signal, str) else args.signal[0]


    #columns_to_exclude = ['dilep_m']
    #signal_df = signal_df.drop(columns=columns_to_exclude, errors='ignore') 
    #background_df = background_df.drop(columns=columns_to_exclude, errors='ignore') 
    #data_df = data_df.drop(columns=columns_to_exclude, errors='ignore')  
    if args.separate_trainings:
        signal_df_low = signal_df[signal_df['dilep_pt'] < 150]
        signal_df_high = signal_df[signal_df['dilep_pt'] >= 150]
        background_df_low = background_df[background_df['dilep_pt'] < 150]
        background_df_high = background_df[background_df['dilep_pt'] >= 150]
        data_df_low = data_df[data_df['dilep_pt'] < 150]
        data_df_high = data_df[data_df['dilep_pt'] >= 150]
        signal_df_low['target'] = 1
        background_df_low['target'] = 0
        signal_df_high['target'] = 1
        background_df_high['target'] = 0
        df_low = pd.concat([signal_df_low, background_df_low], ignore_index=True)   
        df_high = pd.concat([signal_df_high, background_df_high], ignore_index=True) 

        X_low = df_low.drop('target', axis=1)
        y_low = df_low['target']
        smote_low = SMOTE(random_state=42)
        X_low, y_low = smote_low.fit_resample(X_low, y_low)
        X_high = df_high.drop('target', axis=1)
        y_high = df_high['target']
        smote_high = SMOTE(random_state=45)
        X_high, y_high = smote_high.fit_resample(X_high, y_high)
        print(f"After SMOTE, counts of label '1': {sum(y_low == 1)}")
        print(f"After SMOTE, counts of label '0': {sum(y_low == 0)}")
        print(f"After SMOTE, counts of label '1' in high: {sum(y_high == 1)}")
        print(f"After SMOTE, counts of label '0' in high: {sum(y_high == 0)}")

    
    # Add target column to signal and background dataframes
    signal_df['target'] = 1
    background_df['target'] = 0
    
    # Determine the minority class
    minority_class = 'signal' if len(signal_df) < len(background_df) else 'background'

    print(f"Minority class: {minority_class}")

    # Concatenate signal and background dataframes
    df = pd.concat([signal_df, background_df], ignore_index=True)    

    print(f"Length of signal_df: {len(signal_df)}")
    print(f"Length of background_df: {len(background_df)}")
    
    X = df.drop('target', axis=1)
    y = df['target']
    smote = SMOTE(random_state=43)
    X, y = smote.fit_resample(X, y)
    
    print(f"After SMOTE, counts of label '1': {sum(y == 1)}")
    print(f"After SMOTE, counts of label '0': {sum(y == 0)}")

    if args.mode == 'train':
        if args.model_type == 'lgbm':
            if args.separate_trainings:
                model_low = train_model(X_low, y_low, data_df_low, common_signal_name + '_low', args.background)
                model_high = train_model(X_high, y_high, data_df_high, common_signal_name + '_high', args.background)
            else:
                model = train_model(X, y, data_df, common_signal_name, args.background)
        elif args.model_type == 'dnn':
            if args.separate_trainings:
                model_low = train_dnn(X_low, y_low, data_df_low, common_signal_name + '_low', args.background)
                model_high = train_dnn(X_high, y_high, data_df_high, common_signal_name + '_high', args.background)
            else:
                model = train_dnn(X, y, data_df, common_signal_name, args.background)
            
    elif args.mode == 'eval':
        if args.model_type == 'lgbm':
            model = lgb.Booster(model_file=args.model)
        elif args.model_type == 'dnn':
            model = load_model(args.model)
        evaluate_model(model, X, y, data_df, args.signal, args.model_type)

if __name__ == "__main__":
    main()
