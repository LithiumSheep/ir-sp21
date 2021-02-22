import argparse
from itertools import groupby

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


class SegmentClassifier:
    def train(self, trainX, trainY):
        self.clf = DecisionTreeClassifier()  # TODO: experiment with different models
        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)

    def extract_features(self, text):
        words = text.split()
        features = [  # TODO: add features here
            len(text),
            len(text.strip()),
            len(words),

            # '>' seems to represent quoted text
            1 if '>' in words else 0,

            # quoted text seems to always _start_ with '>'
            1 if '>' in words[0] else 0,

            # number of spaces
            text.count(' '),

            # number of capitalized words in line
            sum(1 if w.isupper() else 0 for w in words),

            # number of numeric words in line
            sum(1 if w.isnumeric() else 0 for w in words),

            1 if any(c.isdigit() for c in words[0]) else 0,
            
            # checking for ':'
            1 if ':' in words else 0,

            # and the number of ':' in a line
            sum(1 if ':' else 0 for w in words),

            # different punctuations
            1 if ',' in words or '.' in words else 0,

            1 if '\t' in words else 0,

            1 if '--' in words else 0,
        ]
        return features

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)


def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split('\t', 1)
            if arr[0] == '#BLANK#':
                continue
            X.append(arr[1])
            y.append(arr[0])
        return X, y


def lines2segments(trainX, trainY):
    segX = []
    segY = []
    for y, group in groupby(zip(trainX, trainY), key=lambda x: x[1]):
        if y == '#BLANK#':
            continue
        x = '\n'.join(line[0].rstrip('\n') for line in group)
        segX.append(x)
        segY.append(y)
    return segX, segY


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()

    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)

    if args.format == 'segment':
        trainX, trainY = lines2segments(trainX, trainY)
        testX, testY = lines2segments(testX, testY)

    classifier = SegmentClassifier()
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    if args.errors is not None:
        with open(args.errors, 'w') as fout:
            for y, h, x in zip(testY, outputs, testX):
                if y != h:
                    print(y, h, x, sep='\t', file=fout)

    if args.report:
        print(classification_report(testY, outputs))
    else:
        evaluate(outputs, testY)


if __name__ == '__main__':
    main()