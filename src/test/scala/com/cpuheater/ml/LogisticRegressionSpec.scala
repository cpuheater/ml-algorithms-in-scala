package com.cpuheater.ml

import com.cpuheater.ml.supervised.{LinearRegression, LogisticRegression}
import com.cpuheater.ml.util.TestSupport
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

class LogisticRegressionSpec  extends TestSupport{


  it should "test logistic regression" in {

    val numLinesToSkip = 1
    val delimiter = ','
    val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new ClassPathResource("twodimcat.csv").getFile))
    val iter:DataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000000,2,2, true)
    val dataSet: DataSet = iter.next()


    val x = dataSet.getFeatures()
    val y = dataSet.getLabels()


    val lr = 0.1f
    val iterations = 1000

    val model = new LogisticRegression()
    model.fit(x, y, lr, iterations)

    val point1 = Array(1.6, 7)
    val point2 = Array(7, 1.6)
    val pred1 = model.predict(Nd4j.create(point1))
    val pred2 = model.predict(Nd4j.create(point2))
    println(s"Point${point1.toList} belongs to class ${pred1}")
    println(s"Point${point2.toList} belongs to class ${pred2}")


  }

}
