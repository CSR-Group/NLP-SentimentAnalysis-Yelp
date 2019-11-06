from rnn import main as rnn_main
from ffnn import main as ffnn_main


FLAG = 'RNN'


def main():
	if FLAG == 'RNN':
		hidden_dim1 = 128
		hidden_dim2 = 32
		number_of_epochs = 30
		rnn_main(h1=hidden_dim1, h2=hidden_dim2, number_of_epochs=number_of_epochs)

	elif FLAG == 'FFNN':
		hidden_dim = 32
		number_of_epochs = 10
		ffnn_main(hidden_dim=hidden_dim, number_of_epochs=number_of_epochs)

if __name__ == '__main__':
	main()