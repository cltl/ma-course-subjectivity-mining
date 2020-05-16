import os
import pandas as pd
import shutil

rootDir = "../TRAC2018/"
# input
devInfile = rootDir + "english/agr_en_dev.csv"
testInfile1 = rootDir + "trac-gold-set/agr_en_fb_gold.csv"
testInfile2 = rootDir + "trac-gold-set/agr_en_tw_gold.csv"
trainInfile = rootDir + "english/agr_en_train.csv"
# ouput
vuaDir = "VUA_format/"

trainDataFile = rootDir + vuaDir + "trainData.csv"
testDataFileI = rootDir + vuaDir + "testData-fb.csv"
testDataFileII = rootDir + vuaDir + "testData-tw.csv"
devDataFile = rootDir + vuaDir + "devData.csv"

f1 = 'Id'
f2 = 'Text'
f3 = 'Label'


def readfile(f):
    f = open(f, 'r')  # fout interpretatie:UTF-8, ISO-8859-1,-2,-15, latin1 #foutmelding:Windows-1252, ASCII, UTF-16
    lines = f.readlines()
    return (lines)


def makeDataFile(inFile, DfObj, outputFile):
    df1 = pd.read_csv(inFile, skiprows=0, header=None)  # doctest: +SKIP
    print(
        "\ndataset:{}\tnr of rows:{}\tnr of columns:{}".format(inFile.replace(rootDir, ""), df1.shape[0], df1.shape[1]))

    for i, row in df1.iterrows():
        # print(i)
        cleantweet = df1.loc[i][1].replace("\t", "").replace("\n", "")
        DfObj = DfObj.append({f1: df1.loc[i][0], f2: cleantweet, f3: df1.loc[i][2]}, ignore_index=True)
    print(DfObj.shape)
    print("{} processed lines from {}\t rows/columns`:{} written to {}".format(i + 1, inFile.replace(rootDir, ""),
                                                                               DfObj.shape,
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

    makeDataFile(devInfile, dfDev, devDataFile)
    makeDataFile(trainInfile, dfTrain, trainDataFile)
    makeDataFile(testInfile1, dfTest, testDataFileI)
    makeDataFile(testInfile2, dfTest, testDataFileII)


if __name__ == "__main__":
    main()
