# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import nltk
import os
import math
import string
import sentence
import re
from nltk.corpus import stopwords
from gensim.summarization import keywords
from sklearn.feature_extraction.text import TfidfVectorizer
import pygsheets
import time
from nltk.stem import WordNetLemmatizer


# Create your views here.
@csrf_exempt

def homepage(request):
	return render(request,'project128.html')
@csrf_exempt
def index(request):
	#print "Hello"
	print("In Index view")
	final_summary = ""
	def extract_text(text):
		text_0 =text



		#print 'Text is :-'

		# replace all types of quotations by normal quotes
		text_1 = re.sub("\n"," ",text_0)

		text_1 = re.sub("\"","\"",text_1)
		text_1 = re.sub("''","\"",text_1)
		text_1 = re.sub("``","\"",text_1)

		text_1 = re.sub(" +"," ",text_1)
		return text_1
	def processFile(file_name):

		# read file from provided folder path
		#f = open(file_name,'r')
		text_0 =file_name



		#print 'Text is :-'

		# replace all types of quotations by normal quotes
		text_1 = re.sub("\n"," ",text_0)

		text_1 = re.sub("\"","\"",text_1)
		text_1 = re.sub("''","\"",text_1)
		text_1 = re.sub("``","\"",text_1)

		text_1 = re.sub(" +"," ",text_1)

		text_1 = text_1.replace("<TEXT>","")

		#print text_1
		global article
		article = text_1

		#print 'Keywords in article are : '
		#print keywords(text_1)

		global data1


		data1=keywords(text_1)
		#data1=data1.encode('ascii','ignore')
		data1=data1.replace('\n'," ")
		#print type(data1)
		keyword_reges1=re.compile(r'[\S]+')
		data1= keyword_reges1.findall(data1)



		article_low = text_1.lower()

		#print article_low

		art_occ = 0

		for x in range(0,len(data1)):
			#print data1[x],'-',article_low.count(data1[x])
			art_occ = art_occ + article_low.count(data1[x])

	# 	Testing

		print ("Total Occurences of Keywords in Artice : ")
		print (art_occ)

		global occ
		occ = art_occ
		art_occ = 0

		print ('No of words in articles are : ')
		print (len(text_1.split()))

		print ('No of keywords in articles are : ')
		print (len(data1))


		# segment data into a list of sentences
		sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
		lines = sentence_token.tokenize(text_1.strip())

		# setting the stemmer
		sentences = []
		porter = nltk.PorterStemmer()

		# modelling each sentence in file as sentence object
		for line in lines:

			# original words of the sentence before stemming
			originalWords = line[:]
			line = line.strip().lower()

			# word tokenization
			sent = nltk.word_tokenize(line)

			# stemming words
			stemmedSent = [porter.stem(word) for word in sent]
			stemmedSent = filter(lambda x: x!='.'and x!='`'and x!=','and x!='?'and x!="'"
				and x!='!' and x!='''"''' and x!="''" and x!="'s", stemmedSent)

			# list of sentence objects
			stemmedSent=list(stemmedSent)
			if stemmedSent != []:
				sentences.append(sentence.sentence(file_name, stemmedSent, originalWords))




		return sentences


# Description	: Function to find the term frequencies of the words in the
#				  sentences present in the provided document cluster
	def TFs(sentences):
		# initialize tfs dictonary
		tfs = {}
		wordFreqs = {}

		# for every sentence in document cluster
		for sent in sentences:

			wordFreqs = sent.getWordFreq()
			for word in wordFreqs.keys():
				#print(word)
				if tfs.get(word, 0)!=0:
					tfs[word] = tfs[word] + wordFreqs[word]
				else:
					tfs[word] = wordFreqs[word]
		return tfs


# Description	: Function to find the inverse document frequencies of the words in
#				  the sentences present in the provided document cluster

	def IDFs(sentences):
		N = len(sentences)
		idf = 0
		idfs = {}
		words = {}
		w2 = []
		# every sentence in our cluster
		for sent in sentences:
			# every word in a sentence

			#print(list(sent.getPreProWords()))
			for word in list(sent.getPreProWords()):
				# not to calculate a word's IDF value more than once
				if sent.getWordFreq().get(word, 0) != 0:
					words[word] = words.get(word, 0)+ 1
		# for each word in words
		#print(words)
		for word in words:
			n = words[word]# avoid zero division errors
			try:
				w2.append(n)
				idf = math.log10(float(N)/n)
			except ZeroDivisionError:
				idf = 0
			idfs[word] = idf
		return idfs


# Description	: Function to find TF-IDF score of the words in the document cluster

	def TF_IDF(sentences):
<<<<<<< HEAD
	    # Method variables
>>>>>>> 08d71f7d0bb89b1eeb10c18b7eced8869a5130aa
		tfs = {}
		idfs = {}
		tf_idfs = 0
		tfs = TFs(sentences)
		idfs = IDFs(sentences)
		retval = {}
<<<<<<< HEAD
		for word in tfs:
			tf_idfs=  tfs[word] * idfs[word]
			if retval.get(tf_idfs, None) == None:
				retval[tf_idfs] = [word]
			else:
				retval[tf_idfs].append(word)
		print("retval from tf_idf is")
		print(retval)
		return retval
>>>>>>> 08d71f7d0bb89b1eeb10c18b7eced8869a5130aa
# Description	: Function to find the sentence similarity for a pair of sentences
#				  by calculating cosine similarity

	def sentenceSim(sentence1, sentence2, IDF_w):
		numerator = 0
		denominator = 0
<<<<<<< HEAD
>>>>>>> 08d71f7d0bb89b1eeb10c18b7eced8869a5130aa
		for word in sentence2.getPreProWords():
			numerator+= sentence1.getWordFreq().get(word,0) * sentence2.getWordFreq().get(word,0) *  IDF_w.get(word,0) ** 2

		for word in sentence1.getPreProWords():
			denominator+= ( sentence1.getWordFreq().get(word,0) * IDF_w.get(word,0) ) ** 2

		# check for divide by zero cases and return back minimal similarity
		try:
			return numerator / math.sqrt(denominator)
		except ZeroDivisionError:
			return float("-inf")


	# Description	: Function to build a query of n words on the basis of TF-IDF value


	def buildQuery(sentences, TF_IDF_w, n):
		#sort in descending order of TF-IDF values
		scores = TF_IDF_w.keys()
		#print(scores)
		scores=list(reversed(sorted(scores)))
		#print("after")
		#print(scores
		i = 0
		j = 0
		queryWords = []

		# select top n words
		while(i<n):
			words = TF_IDF_w[scores[j]]
			for word in words:
				queryWords.append(word)
				i=i+1
				if (i>n):
					break
			j=j+1


		# return the top selected words as a sentence
		return sentence.sentence("query", queryWords, queryWords)


# Description	: Function to find the best sentence in reference to the query

	def bestSentence(sentences, query, IDF):
		best_sentence = None
		maxVal = float("-inf")

		for sent in sentences:
			similarity = sentenceSim(sent, query, IDF)

			if similarity > maxVal:
				best_sentence = sent
				maxVal = similarity
		sentences.remove(best_sentence)



		return best_sentence


# Description	: Function to create the summary set of a desired number of words
	def makeSummary(sentences, best_sentence, query, summary_length, lambta, IDF):
		summary = [best_sentence]
		sum_len = len(best_sentence.getPreProWords())

		MMRval={}

		# keeping adding sentences until number of words exceeds summary length
		while (sum_len < summary_length):
			MMRval={}
			for sent in sentences:
				MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)

			maxxer = max(MMRval, key=MMRval.get)
			summary.append(maxxer)
			sentences.remove(maxxer)
			sum_len += len(maxxer.getPreProWords())
		return summary


# Description	: Function to calculate the MMR score given a sentence, the query
#				  and the current best set of sentences

	def MMRScore(Si, query, Sj, lambta, IDF):
		Sim1 = sentenceSim(Si, query, IDF)
		l_expr = lambta * Sim1
		value = [float("-inf")]

		for sent in Sj:
			Sim2 = sentenceSim(Si, sent, IDF)
			value.append(Sim2)

		r_expr = (1-lambta) * max(value)
		MMR_SCORE = l_expr - r_expr



		return MMR_SCORE

# -------------------------------------------------------------
#	MAIN FUNCTION




	start_time=time.time()
	input = request.POST.get('message')
	sentences = []
	#print type(input)
	#print "Incoming"
	print(type(input))
	import numpy as np
	np.random.seed(42)
	sentences = sentences + processFile(input)
	#print("before")
	#print(sentences[0].getPreProWords())



	import numpy as np
	cv=TfidfVectorizer(min_df=1,stop_words='english')
	traincv=cv.fit_transform([input])
	scores=zip(cv.get_feature_names(),np.asarray(traincv.sum(axis=0)).ravel())

	sorted_scores=sorted(scores,key=lambda x:x[1],reverse=True)
	#print sorted_scores
	#print 'Get feature names for the document '
	from pprint import pprint
	#pprint(sorted_scores[:15])
	#keywords_tfidf=sorted_scores[:10]




	# calculate TF, IDF and TF-IDF scores
	#TF_w 		= TFs(sentences)
	IDF_w 		= IDFs(sentences)
	TF_IDF_w 	= TF_IDF(sentences)

	# build query; set the number of words to include in our query
	query = buildQuery(sentences, TF_IDF_w, 10)
	#global a
	# pick a sentence that best matches the query
	best1sentence = bestSentence(sentences, query, IDF_w)
	'''vectorizer=TfidfVectorizer(min_df=1,stop_words='english')

	model=vectorizer.fit_transform([input])
	print 'using tf-idf vectorizer'
	data= vectorizer.vocabulary_.items()
	print data
	'''
	# build summary by adding more relevant sentences
	summary = makeSummary(sentences, best1sentence, query, 100, 0.5, IDF_w)

	global final_summary
	final_summary = ''




	for sent in summary:
		final_summary = final_summary + sent.getOriginalWords() + "\n"
	final_summary = final_summary[:-1]
	#results_folder = os.getcwd() + "/MMR_results"
	#with open(os.path.join(results_folder,(str(folder) + ".MMR")),"w") as fileOut: fileOut.write(final_summary)
	#print "The data type is ",type(final_summary)

	# for fiding keywords and their occurrences
	final_summary = final_summary.encode('utf-8')
	#final_summary = final_summary.replace("\n"," ")
	print (final_summary)
	#final_summary = re.sub("<TEXT> ",'',final_summary)
	#final_summary = re.sub("<TEXT>",'',final_summary)
	#print input
	#print '\n\n'
	#print final_summary
	#print 'Keywords in summary are :  '
	#print keywords(final_summary)
	data=keywords(final_summary,ratio=0.2)
	data=data.encode('ascii','ignore')
	#data=data.replace('\n'," ")
	#print 'data before regex',data
	#keyword_reges=re.compile(r'[\S]+')
	#data= keyword_reges.findall(data)

	'''
	#Using minimum of sentencesx2 and 20
	no_of_sentence = final_summary.split(".")
	print 'Printing list of sentences : '
	print no_of_sentence
	print len(no_of_sentence)
	print min(len(no_of_sentence),20)
	'''

	#Setting local threshold
	article_length =  len(article.split())
	threshold = 0

	if article_length <= 300:
		threshold = 9
	elif article_length >300 and article_length <=400:
		threshold = 13
	elif article_length >400 and article_length <=500:
		threshold = 16
	elif article_length >500 and article_length <=600:
		threshold = 19
	elif article_length >600 and article_length <=700:
		threshold = 22
	elif article_length >700 and article_length <=800:
		threshold = 21
	elif article_length >800 and article_length <=1000:
		threshold = 31
	elif article_length >1000 and article_length <=1100:
		threshold = 39
	elif article_length >1100 and article_length <=1600:
		threshold = 45
	elif article_length >1600 and article_length <=1700:
		threshold = 64


	keywords_tfidf=[x[0].encode('utf-8') for x in sorted_scores[:threshold]]
	#print keywords_tfidf




	print ("List of Keywords : ")
	#print data
	print (keywords_tfidf)


	summary_low = final_summary.lower()

	sum_occ = 0

	for x in range(0,len(keywords_tfidf)):
		#print keywords_tfidf[x],'-',summary_low.count(keywords_tfidf[x])
		sum_occ = sum_occ + summary_low.count(keywords_tfidf[x])

	print ("Total Occurrences of Keywords in Summary : ")
	print (sum_occ)




	#summary keyword length
	print ('Summary Keyword Length : ')
	#print len(data)
	print (len(keywords_tfidf))

	print ('No of words in summary are : ')
	print (len(final_summary.split()))

	response_time = (time.time()-start_time)
	print ('The response time is  : ',response_time)



	'''
	#SPREADSHEETS
	import gspread
	from oauth2client.service_account import ServiceAccountCredentials


	# use creds to create a client to interact with the Google Drive API
	scope = ['https://spreadsheets.google.com/feeds']
	creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
	client = gspread.authorize(creds)

	# Find a workbook by name and open the first sheet
	# Make sure you use the right name here.
	sheet = client.open("project128").sheet1

	row = ["I'm","inserting","a","row","into","a,","Spreadsheet","with","Python"]
	index = 2
	sheet.insert_row(row, index)


	import pygsheets



	gc = pygsheets.authorize(service_file='client_secret.json')
	sh = gc.open('project128')
	wks = sh.sheet1

	# Update a cell with value (just to let him know values is updated ;) )
	wks.update_cell('A1', "Hey yank this numpy array")
	'''




	#CSV
	import csv
	fields=[article,final_summary,len(article.split()),len(final_summary.split()),data1,keywords_tfidf,len(data1),len(keywords_tfidf),occ,sum_occ, response_time]
	with open('stats.csv', 'a') as f:
		 writer = csv.writer(f)
		 writer.writerow(fields)
	sum_occ = 0
	return HttpResponse(final_summary)
