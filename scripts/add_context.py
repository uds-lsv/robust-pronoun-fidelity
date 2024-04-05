import csv
import sys
import itertools
from pronouns import mapping
from pathlib import Path
import pandas as pd

def instantiate_template(template, occupation, pronoun_type, pronoun):
    return template.replace('$OCCUPATION/PARTICIPANT', occupation).replace(pronoun_type, pronoun)

def build_pronoun_type_template_mapping(filename):
    pronoun_type_template_mapping = {
        'explicit_template': {pronoun_type: [] for pronoun_type in mapping},
        'implicit_template': {pronoun_type: [] for pronoun_type in mapping},
    }

    df = pd.read_csv(filename, delimiter='\t')
    for i, row in df.iterrows():
        pronoun_type = row['pronoun_type']
        polarity = row['polarity']
        for key in pronoun_type_template_mapping:
            pronoun_type_template_mapping[key][pronoun_type].append((row[key], polarity))

    return pronoun_type_template_mapping

def get_output_line(row, context, pronoun1, uid, confuse=''):
    capitalized = [c.capitalize() for c in context]
    template = ' '.join((*capitalized, row['sentence']))
    return '\t'.join([row['occupation'],
                      row['participant'],
                      template,
                      row['pronoun_type'],
                      row['word'],
                      pronoun1,
                      uid,
                      confuse]) + '\n'

def add_context(filename, pronoun_type_template_mapping, occupation):
    f = 'o' if occupation else 'p' # first
    s = 'p' if occupation else 'o' # second
    basename = Path(filename).stem
    with open(filename, 'r', encoding='utf-8') as in_f, \
         open(f'e{f}_{basename}.tsv', 'w', encoding='utf-8') as ef_f, \
         open(f'e{f}_e{s}_{basename}.tsv', 'w', encoding='utf-8') as ef_es_f, \
         open(f'e{f}_e{s}_i{s}_{basename}.tsv', 'w', encoding='utf-8') as ef_es_is_f, \
         open(f'e{f}_e{s}_i{s}_i{s}_{basename}.tsv', 'w', encoding='utf-8') as ef_es_is_is_f, \
         open(f'e{f}_e{s}_i{s}_i{s}_i{s}_{basename}.tsv', 'w', encoding='utf-8') as ef_es_is_is_is_f, \
         open(f'e{f}_e{s}_i{s}_i{s}_i{s}_i{s}_{basename}.tsv', 'w', encoding='utf-8') as ef_es_is_is_is_is_f:
        header = 'occupation\tparticipant\tsentence\tpronoun_type\tword\tpronoun\tuid\tconfuse_pronoun\n'
        ef_f.write(header)
        ef_es_f.write(header)
        ef_es_is_f.write(header)
        ef_es_is_is_f.write(header)
        ef_es_is_is_is_f.write(header)
        ef_es_is_is_is_is_f.write(header)
        reader  = csv.DictReader(in_f, delimiter='\t')
        first = 'occupation' if occupation else 'participant'
        second = 'participant' if occupation else 'occupation'
        for row in reader:
            pronoun_type = row['pronoun_type']
            pronouns = mapping[pronoun_type]
            for i, (e1, s1) in enumerate(pronoun_type_template_mapping['explicit_template'][pronoun_type]):
                for pronoun1 in pronouns:
                    intro1 = instantiate_template(e1, row[first], pronoun_type, pronoun1)
                    ef_f.write(get_output_line(row, [intro1], pronoun1, f'e{f}{i}'))

                    for j, (e2, s2) in enumerate(pronoun_type_template_mapping['explicit_template'][pronoun_type]):
                        if (j % 5) == (i % 5): # second template cannot have the same content as the first, regardless of polarity
                            continue
                        if s2 == s1: # use the opposite sentiment
                            continue
                        for pronoun2 in pronouns:
                            if pronoun1 == pronoun2: # we need unique pronouns for each entity being spoken about
                                continue
                            intro2 = instantiate_template(e2, row[second], pronoun_type, pronoun2)
                            ef_es_f.write(get_output_line(row, [intro1, intro2], pronoun1, f'e{f}{i}_e{s}{j}', pronoun2))

                            # implicit continuations must have the same sentiment and referent as the last intro
                            # it should not have the same content as either intro
                            # i.e., there should be 4 options
                            implicit_continuations = []
                            for k, (it, st) in enumerate(pronoun_type_template_mapping['implicit_template'][pronoun_type]):
                                if k == j or k == i:
                                    continue
                                if st != s2:
                                    continue
                                # must be filled with the same pronoun as the last intro because it is the same referent
                                implicit = instantiate_template(it, row[second], pronoun_type, pronoun2)
                                implicit_continuations.append((k, implicit))
                            assert len(implicit_continuations) == 4

                            for perm in itertools.permutations(implicit_continuations, 1):
                                k1, i1 = perm[0]
                                ef_es_is_f.write(get_output_line(row, [intro1, intro2, i1], pronoun1,
                                                 f'e{f}{i}_e{s}{j}_i{s}{k1}', pronoun2))

                            for perm in itertools.permutations(implicit_continuations, 2):
                                k1, i1 = perm[0]
                                k2, i2 = perm[1]
                                ef_es_is_is_f.write(get_output_line(row, [intro1, intro2, i1, i2], pronoun1,
                                                 f'e{f}{i}_e{s}{j}_i{s}{k1}_i{s}{k2}', pronoun2))

                            # exploit the fact that perm(S, 3) == perm(S, 4) when |S| == 4
                            for perm in itertools.permutations(implicit_continuations, 4):
                                k1, i1 = perm[0]
                                k2, i2 = perm[1]
                                k3, i3 = perm[2]
                                k4, i4 = perm[3]
                                ef_es_is_is_is_f.write(get_output_line(row, [intro1, intro2, i1, i2, i3], pronoun1,
                                                 f'e{f}{i}_e{s}{j}_i{s}{k1}_i{s}{k2}_i{s}{k3}', pronoun2))
                                ef_es_is_is_is_is_f.write(get_output_line(row, [intro1, intro2, i1, i2, i3, i4], pronoun1,
                                                 f'e{f}{i}_e{s}{j}_i{s}{k1}_i{s}{k2}_i{s}{k3}_i{s}{k4}', pronoun2))


def main():
    assert len(sys.argv) == 3
    pronoun_type_template_mapping = build_pronoun_type_template_mapping(sys.argv[2])
    add_context(sys.argv[1], pronoun_type_template_mapping, occupation=True)

if __name__ == '__main__':
    main()
