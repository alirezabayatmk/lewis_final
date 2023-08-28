import re
import pandas as pd


def text_preprocess(text: str):
    text = text.lower()
    text = text.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    url_pattern = re.compile(r"http[s]?://\S+")
    text = url_pattern.sub(r"<URL>", text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    return text


soft_df = pd.read_csv('raw_datasets/train.csv')
soft_df = soft_df.drop(['post_id'], axis=1)
soft_df = soft_df[['issue','post_text','Inappropriateness','fold0.0']]
soft_df = soft_df.rename(columns={'post_text':'text','Inappropriateness':'label','fold0.0':'fold'})
soft_df['label'] = soft_df['label'].apply(lambda x: 1 if x > 0.5 else 0)


df = pd.read_csv('raw_datasets/appropriateness_corpus_conservative_w_folds.csv')
selected_df = df[['issue', 'post_text','Inappropriateness', 'fold0.0']]
selected_df = selected_df.rename(columns={'post_text': 'text', 'Inappropriateness': 'label', 'fold0.0': 'fold'})

merged_df = pd.concat([soft_df, selected_df], ignore_index=True)

# number of duplicates
print('Number of duplicates: ', merged_df.duplicated().sum())
print('Number of duplicates in text column: ', merged_df.duplicated(subset=['text']).sum())
print('Number of duplicates in both text and issue columns: ', merged_df.duplicated(subset=['text', 'issue']).sum())

# drop duplicates
merged_df = merged_df.drop_duplicates()
merged_df = merged_df.drop_duplicates(subset=['text'])
merged_df = merged_df.drop_duplicates(subset=['text', 'issue'])

# number of duplicates
print('Number of duplicates after dropping: ', merged_df.duplicated().sum())
print('Number of duplicates in text column after dropping: ', merged_df.duplicated(subset=['text']).sum())
print('Number of duplicates in both text and issue columns after dropping: ', merged_df.duplicated(subset=['text', 'issue']).sum())

# number of classes per fold in merged_df
print('Number of classes per fold in merged_df:', merged_df.groupby('fold')['label'].value_counts())

# randomly drop 20000 rows where label is 1 and fold is TRAIN 
merged_df = merged_df.drop(merged_df[(merged_df['label'] == 1) & (merged_df['fold'] == 'TRAIN')].sample(n=23000, random_state=1).index)

# number of classes per fold in merged_df
print('Number of classes per fold in merged_df after sampling:', merged_df.groupby('fold')['label'].value_counts())

merged_df['text'] = merged_df['text'].apply(text_preprocess)


train_df = merged_df[merged_df['fold'] == 'TRAIN']
train_df = train_df[['text', 'label']].reset_index(drop=True)

test_df = merged_df[merged_df['fold'] == 'TEST']
test_df = test_df[['text', 'label']].reset_index(drop=True)

valid_df = merged_df[merged_df['fold'] == 'VALID']
valid_df = valid_df[['text', 'label']].reset_index(drop=True)


# sample size
print('Train size: ', len(train_df))
print('Test size: ', len(test_df))
print('Valid size: ', len(valid_df))

def data_sep(df, name='train'):
    df_pos = df[df['label'] == 1].reset_index(drop=True)
    df_neg = df[df['label'] == 0].reset_index(drop=True)

    with open('downloaded_data/cmv/dm1-app/' + name + '.txt', 'w') as f:
        for i in range(len(df_pos)):
            f.write(df_pos['text'][i] + '\n')

    with open('downloaded_data/cmv/dm2-inapp/' + name + '.txt', 'w') as f:
        for i in range(len(df_neg)):
            f.write(df_neg['text'][i] + '\n')

data_sep(train_df, 'train')
data_sep(test_df, 'test')
data_sep(valid_df, 'valid')