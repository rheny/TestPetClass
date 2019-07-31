#! /usr/bin/env python3
# 121212124s
import sys
total = 0
lastword = None

for line in sys.stdin:
    line = line.strip()

    # recuperer la cle et la valeur et conversion de la valeur en int
    word, count = line.split()
    count = int(count)

    # passage au mot suivant (plusieurs cles possibles pour une même exécution de programme)
    if lastword is None:
        lastword = word
    if word == lastword:
        total += count
    else:
        print("%s\t%d occurences" % (lastword, total))
        total = count
        lastword = word
        
if lastword is not None:
    print("%s\t%d occurences C:\Users\arheny\Downloads\python\petclass" % (lastword, total))


	df = spark.createDataFrame(
		[(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
		["id", "category"])

	indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
	indexed = indexer.fit(df).transform(df)
	indexed.show()
    Row(features=DenseVector
    numpy.ndarray
    TypeError: 'DataFrame' object is not callable