from sklearn.utils import shuffle
import language_tool
import sys
import os

def main():
	path = '/home/lguarise/Desktop/News_corpus/'

	for directory in os.listdir(path):
		print(directory)
		directory_path = os.path.join(path, directory)
		if os.path.isdir(directory_path):
			for filename in os.listdir(directory_path):
				with open(os.path.join(directory_path, filename)) as file:
					for line in file:
						if 'os elementos probatórios do elo' in line.lower():
							print (filename)
							print("*******AQUI*************")
				file.close()

if __name__ == '__main__':
    main()
