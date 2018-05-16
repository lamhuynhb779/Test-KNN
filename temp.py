def ranDom5000doc(path, docs):
	temp = set()
	for f1 in glob.glob(path):
		elements = random.randint(0, 1000)
		if len(temp) < 5000:
			x = set(random.sample(os.listdir(f1), elements))
			temp.update(x)
		else:
			break
	docs.extend(random.sample(temp, 5000))

def chooseNDocFrom15000(path, doc5000, docN, n):
	temp = set()
	while len(temp) < n:
		f1 = path+"/"+random.choice(os.listdir(path))
		elements = random.randint(0, 1000)
		x = set(random.sample(os.listdir(f1), elements)) - set(doc5000)
		temp.update(x)
	docN.extend(random.sample(temp, n))

	
doc5000 = []
docN = []

# print(os.listdir('20_newsgroups/alt.atheism'))

# ranDom5000doc("20_newsgroups/*", doc5000)
# chooseNDocFrom15000("20_newsgroups", doc5000, docN, 1000)