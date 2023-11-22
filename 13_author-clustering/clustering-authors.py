"""
Script to perform clustering on a Term-Document-Matrix using sklearn. 
https://scikit-learn.org/stable/modules/clustering.html
https://scikit-learn.org/stable/modules/clustering.html#k-means
"""


"""
Aufgabenbeschreibung:
 
Die Aufgabe behandelt das Clustering von 30 englischen Romanen von 10 verschiedenen Autor:innen. 
Die Daten liegen bereits als Term-Dokument-Matrix vor: Für jeden Roman sind das die relativen Häufigkeiten der häufigsten 5000 Wörter (ELTeC-eng_TDM.csv).
Das Ziel ist es, jeden Roman einem Cluster zuzuordnen. Idealerweise würden die Cluster mit den Autorennamen übereinstimmen. 
Berichtet werden soll die Qualität des Clusterings (als Cluster Purity und/oder Rand Index.)
Gehen Sie wie folgt vor: 
- Laden Sie die Term-Dokument-Matrix. 
- Verwenden Sie den kMeans-Algorithmus aus sklearn. 
- Geben Sie für jeden Roman den Autornamen und die Cluster-Zuordnung an.
- Berechnen Sie die Qualität des Clusterings, beispielsweise als Cluster Purity.
"""
