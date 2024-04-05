from glob import glob
import pandas as pd

for f in glob('*.tsv'):
    df = pd.read_csv(f, sep='\t')
    occupations_only = df[df.occupation == df.word]
    print(f, len(occupations_only))

    for seed in [13, 17, 19]:
        if len(occupations_only) == 7200:
            sampled = occupations_only.groupby(['word', 'pronoun_type', 'pronoun']).sample(3, random_state=seed)
        else:
            sampled = occupations_only.groupby(['word', 'pronoun_type', 'pronoun', 'confuse_pronoun']).sample(1, random_state=seed)
        sampled.to_csv(f'{seed}_{f}', sep='\t', index=None)
