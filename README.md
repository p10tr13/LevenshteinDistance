Instrukcja obsługi programu

Podstawowe wywołanie programu: s1_s2_txt_file_path txt_output_file_path gdzie:

s1_s2_txt_file_path - to plik .txt, w którym w pierwszej linii jest s1, a w drugiej s2  
txt_output_file_path - scieżka do pliku, gdzie mają być zapisane operacje przekształcania słowa s1 na s2

Można jeszcze wywołać program w poniższy sposób:

Prawidłowe argumenty: zaawansowany_tryb_programu arg2 arg3 (optional) print_mode  
Przykład: 1 ala lal 3

Zaawansowane tryby pracy:
1. dwa słowa z liter z przedziału 'A' - 'Z'
2. dwie liczby dodatnie większe od 2 (program losuje litery do tych dwóch słów o podanej dlugości)

Opcjonalnie po argumentach trybu można dodać sposób wypisania wyniku:
1. Wypisanie na konsole tabeli D (tylko GPU)
2. Wypisanie na konsole listy zamian s1 na s2 (tylko GPU)
3. Zapisanie do plików tabel D z CPU i GPU
4. Tryb wypisywania 1 i 2
5. Tryb wypisywania 2 i 3
6. Tryb 1, 2 i 3

Dodatkowe informacje:
Długość s2 jest ograniczona do ilości SM w GPU razy 1024 
