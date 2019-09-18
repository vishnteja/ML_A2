import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import random
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score,accuracy_score

category = []
reviews = []
document_id = []
document = []
unique_words = []
for line in open('naive_bayes_data.txt','r', encoding='utf-8-sig'):	
    a = line.split()
    category.append(a[0])
    reviews.append(a[1])
    document_id.append(a[2])
    document.append(" ".join(a[3:]))
random.shuffle(document)
no_of_docs = len(document)
train = document[:int(0.8*(no_of_docs))]
test = document[int(0.8*(no_of_docs)):]
true_reviews = reviews[int(0.8*(no_of_docs)):]
train_len = len(train)
stop_words1 = set(stopwords.words('english'))
stop_words2 = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]
for doc in train:
	words = doc.split() 
	words_cleaned = [w.lower() for w in words if w not in stop_words1 and w not in stop_words2 and w[0] not in punctuation and not any(c.isdigit() for c in w) and w.isalpha()]
	unique_words.extend(words_cleaned)
vocabulary = sorted(list(set(unique_words)))
d = len(vocabulary)
a = 1
count = {key : [0,0] for key in vocabulary}
negdocs = []
posdocs = []
for i in range(no_of_docs):
	if(reviews[i] == 'neg'):
		negdocs.append(i)
	else:
		posdocs.append(i)

for i in negdocs:
	words = document[i].split()
	words_cleaned = [w.lower() for w in words if w not in stop_words1 and w not in stop_words2 and w[0] not in punctuation and not any(c.isdigit() for c in w) and w.isalpha()]
	for word in set(words_cleaned):
		try:
			count[word][0] = count[word][0] + 1
		except KeyError:
			count[word] = [0,0]
			count[word][0] = count[word][0] + 1

for i in posdocs:
	words = document[i].split()
	words_cleaned = [w.lower() for w in words if w not in stop_words1 and w not in stop_words2 and w[0] not in punctuation and not any(c.isdigit() for c in w) and w.isalpha()]
	for word in set(words_cleaned):
		try:
			count[word][1] = count[word][1] + 1
		except KeyError:
			count[word] = [0,0]
			count[word][1] = count[word][1] + 1

result = []
for doc in test:
	words = doc.split() 
	words_cleaned = [w.lower() for w in words if w not in stop_words1 and w not in stop_words2 and w[0] not in punctuation and not any(c.isdigit() for c in w) and w.isalpha()]
	score0 = 1
	score1 = 1
	for word in words_cleaned:
		try:
			score0 = score0*(((count[word][0]) / (len(negdocs))))
			score1 = score1*(((count[word][1]) / (len(posdocs))))
		except KeyError:
			score0 = score0 * 0.0001
			score1 = score1*0.0001
	score0 = score0 * (len(negdocs) / no_of_docs)
	score1 = score1 * (len(posdocs) / no_of_docs)

	if(score0 > score1):
		result.append('neg')
	else:
		result.append('pos')

print("\nAccuracy: " + str(accuracy_score(true_reviews,result)))

conf_matrix = confusion_matrix(true_reviews,result,labels=['pos','neg'])
print("\nConfusion Matrix: ")
print("               Predicted")
print("               pos   neg")
print("Actual pos    ",end="")
print(conf_matrix[0])
print("       neg    ",end="")
print(conf_matrix[1])
print("\nPrecision")
print("pos: " + str(precision_score(true_reviews,result,pos_label = "pos")))
print("neg: " + str(precision_score(true_reviews,result,pos_label = "neg")))

print("\nRecall")
print("pos: " + str(recall_score(true_reviews,result,pos_label = "pos")))
print("neg: " + str(recall_score(true_reviews,result,pos_label = "neg")))

print("\nF1 Measure")
print("pos: " + str(f1_score(true_reviews,result,pos_label = "pos")))
print("neg: " + str(f1_score(true_reviews,result,pos_label = "neg")))
