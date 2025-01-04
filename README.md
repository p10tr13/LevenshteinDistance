Instrukcja obslugi programu

Podstawowe wywo³anie programu: s1_s2_txt_file_path txt_output_file_path gdzie:

s1_s2_txt_file_path - to plik .txt, w ktorym w pierwszej linii jest s1, a w drugiej s2
txt_output_file_path - sciezka do pliku, gdzie maja byc zapisane operacje przeksztalcania slowa s1 na s2

Mozna jeszcze wywolac program w ponizszy sposob:

Prawidlowe argumenty: zaawansowany_tryb_programu arg2 arg3 (optional) print_mode
Przyklad: 1 ala lal 3

Zaawansowane tryby pracy:
1. dwa slowa z liter z przedzialu 'A' - 'Z'
2. dwie liczby dodatnie wieksze od 2 (program losuje litery do tych dwoch slow o podanej dlugosci)

Opcjonalnie po argumentach trybu mozna dodac sposob wypisana wyniku:
1. Wypisanie na konsole tabeli D (tylko GPU)
2. Wypisanie na konsole listy zamian s1 na s2 (tylko GPU)
3. Zapisanie do plikow tabel D z CPU i GPU
4. Tryb wypisywania 1 i 2
5. Tryb wypisywania 2 i 3
6. Tryb 1, 2 i 3

Dodatkowe informacje:
dlugosc s2 jest ograniczona do iloœci SM w GPU razy 1024 