import pickle
import os

def inspect(path):
    print('Inspecting:', path)
    if not os.path.exists(path):
        print('  File not found')
        return
    obj = pickle.load(open(path, 'rb'))
    print('  Type:', type(obj))
    # common sklearn attribute
    print('  n_features_in_:', getattr(obj, 'n_features_in_', None))
    # if it's a pipeline, show steps
    for attr in ('classes_', 'coef_', 'feature_count_'):
        if hasattr(obj, attr):
            print(' ', attr, 'present')

if __name__ == '__main__':
    for p in ('MNB.pkl', 'RFC.pkl', 'DTC.pkl'):
        inspect(p)
