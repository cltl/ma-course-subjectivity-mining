import os
import pandas as pd
import shutil

rootDir = "../hate-speech-dataset-vicom/"
# input
devDir = rootDir + "sampled_train/"
testDir = rootDir + "sampled_test/"
trainDir = rootDir + "all_files/"
metadataFile = rootDir + "annotations_metadata.csv"
# ouput
vuaDir = "VUA_format/"
trainDataFile = rootDir + vuaDir + "trainData.csv"
testDataFile = rootDir + vuaDir + "testData.csv"
devDataFile = rootDir + vuaDir + "devData.csv"

f1 = 'Id'
f2 = 'Text'
f3 = 'Label'


def readfile(f):
    f = open(f, 'r')  # fout interpretatie:UTF-8, ISO-8859-1,-2,-15, latin1 #foutmelding:Windows-1252, ASCII, UTF-16
    lines = f.readlines()
    return (lines)


def makeDataFile(Dir, df1, DfObj, outputFile):
    fcounter = 0
    for (dir, _, files) in os.walk(Dir):
        for f in files:
            fcounter += 1

            lines = readfile(Dir + f)
            text = ""
            for l in lines:
                if "\t" in l:
                    l = l.replace("\t", "TAB")
                text = text + l
            file_id = f.replace(".txt", "")
            DfObj = DfObj.append({f1: file_id, f2: text, f3: df1.loc[df1['file_id'] == file_id]['label'].item()},
                                 ignore_index=True)
    print("{} processed files\t rows/columns`:{} written to {}".format(fcounter, DfObj.shape,
                                                                       outputFile.replace(rootDir, "")))
    DfObj.to_csv(outputFile, index=False, header=True, sep='\t')


def main():
    mydir = rootDir + vuaDir
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    os.mkdir(mydir)

    dfTrain = pd.DataFrame(columns=[f1, f2, f3])
    dfTest = pd.DataFrame(columns=[f1, f2, f3])
    dfDev = pd.DataFrame(columns=[f1, f2, f3])

    df1 = pd.read_csv(metadataFile, skiprows=0)  # doctest: +SKIP
    print("dataset:{}\tnr of rows:{}\tnr of columns:{}".format(metadataFile, df1.shape[0], df1.shape[1]))

    makeDataFile(devDir, df1, dfDev, devDataFile)
    makeDataFile(testDir, df1, dfTest, testDataFile)
    makeDataFile(trainDir, df1, dfTrain, trainDataFile)


if __name__ == "__main__":
    main()
