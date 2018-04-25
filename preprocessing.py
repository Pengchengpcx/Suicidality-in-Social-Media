# prerpocessing
import datetime
import csv

start = 1
end = 1

post_fields = ["post_id", "user_id", "date", "time", "subreddit", "post_title", "post_body"]

def processline(writer, line):
	line_info 	= line.split("\t")
	if(len(line_info) < 5): 
		print(line_info[0])
		return
	elif (len(line_info) == 5):
		post_id 	= line_info[0]
		user_id 	= line_info[1]
		timestamp 	= datetime.datetime.fromtimestamp(int(line_info[2])).strftime('%Y-%m-%d %H:%M:%S')
		[date,time] = timestamp.split(" ")
		subreddit 	= line_info[3]
		post_title 	= line_info[4]
		post_body 	= ""
	else:
		post_id 	= line_info[0]
		user_id 	= line_info[1]
		timestamp 	= datetime.datetime.fromtimestamp(int(line_info[2])).strftime('%Y-%m-%d %H:%M:%S')
		[date,time] = timestamp.split(" ")
		subreddit 	= line_info[3]
		post_title 	= line_info[4]
		post_body 	= line_info[5]

	writer.writerow({
		"post_id"	: post_id,
		"user_id"	: user_id,
		"date"		: date,
		"time"		: time,
		"subreddit"	: subreddit,
		"post_title": post_title,
		"post_body"	: post_body
	})

if __name__ == "__main__":
	writeFileName = 'post_' + str(start) + '_' + str(end) + '.csv'
	writeFile = open(writeFileName,'w')
	writer = csv.DictWriter(writeFile, delimiter = ',', fieldnames = post_fields)
	writer.writeheader()

	for i in range(start, end + 1):
		print(i)
		readFile = 'sw_users/'+ str(i) + '.posts'
		with open(readFile, mode = 'r') as infile:
			for line in infile:
				processline(writer, line)