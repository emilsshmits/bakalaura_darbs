import PySimpleGUI as sg
from sympy import Symbol, LessThan, Piecewise, solve, Eq, Piecewise, GreaterThan, Max
import sympy
import numpy as np
import pulp

from itertools import chain
from sympy.parsing.sympy_parser import parse_expr


sg.theme("DefaultNoMoreNagging")

# ===================================================
# ===================================================
# Vispirms definē vairākas palīgfunkcijas
# ===================================================
# ===================================================

# ===================================================
# Funkcija, kas palaiž logu, kurā ievada uzdevumu
# ===================================================
def ievadit_uzd(x_nr, funkc_nr, ierob_nr, x_pulp, x_sympy, f_pulp, f_sympy, strikti, alfa):
    funkc_nr_dal_2 = int(funkc_nr / 2)
    
    ierobezojumi = [f"ierob_{i+1}" for i in range(ierob_nr)]
    
    constraints = []
    constraints_sympy = []
    signs = ["=", "<=", ">="]
    
    if strikti:
        merka_funkc = [f"f_{i+1}" for i in range(funkc_nr)]
        # izveidot layoutus priekš mērķa funkcijām
        for i in range(funkc_nr):
            input_koefs = [sg.In(size = (3,1)) for _ in range(len(x_pulp))]
            main = [sg.Text(f"*{x} +") if i+1 != len(x_pulp) else sg.Text(f"*{x}") for i, x in enumerate(x_pulp)]
            merka_funkc[i] = list(chain(*[(kf, mn) for kf, mn in zip(input_koefs, main)]))
    else:
        merka_funkc = [f"f_{i+1}" for i in range(funkc_nr_dal_2)]
        input_koefs_funkc = [[] for _ in range(len(f_pulp))]
        # izveidot layoutus priekš mērķa funkcijām
        for i in range(funkc_nr_dal_2):
            for j in range(x_nr):
                input_koefs_funkc[i].append([[sg.Text("0                         , ja                  x <"), sg.In(size = (3,1))],
                                  [sg.In(size = (3,1)), sg.Text("*x+"), sg.In(size = (3,1)), sg.Text(" , ja "), sg.In(size = (3,1)), sg.Text("<= x <="), sg.In(size = (3,1))],
                                  [sg.In(size = (3,1)), sg.Text("*x+"), sg.In(size = (3,1)), sg.Text(" , ja "), sg.In(size = (3,1)), sg.Text("<= x <="), sg.In(size = (3,1))],
                                  [sg.Text("0                         , ja "), sg.In(size = (3,1)), sg.Text("< x")],])
                input_koefs_funkc[i].append([[sg.Text(f"*{x_pulp[j]} +") if j+1 != len(x_pulp) else sg.Text(f"*{x_pulp[j]}")]])
                
            pagaidu = [sg.Column(col) for col in input_koefs_funkc[i]]
            merka_funkc[i] = []

            for v, skaitlis in enumerate(pagaidu):
                if v % 2 == 0:
                    merka_funkc[i].append(sg.VerticalSeparator())
                    merka_funkc[i].append(skaitlis)
                    merka_funkc[i].append(sg.VerticalSeparator())
                else:
                    merka_funkc[i].append(skaitlis)

            if i != 0:
                merka_funkc[i].append([sg.HorizontalSeparator()])
    
    
    # izveido layoutus priekš ierobežojumiem
    for i in range(ierob_nr):
        input_koefs = [sg.In(size = (3,1)) for _ in range(len(x_pulp))]
        main = [sg.Text(f"*{x} +") if i+1 != len(x_pulp) else sg.Text(f"*{x}") for i, x in enumerate(x_pulp)]
        ierobezojumi[i] = list(chain(*[(kf, mn) for kf, mn in zip(input_koefs, main)]))
        ierobezojumi[i].append(sg.Combo(signs, readonly = True))
        ierobezojumi[i].append(sg.In(size = (3,1)))
        
    layout = [
        [sg.Text("Ievadiet mērķa funkcijas:")],
        merka_funkc,
        [sg.Text("Ievadiet ierobežojumus:")],
        ierobezojumi,
        [sg.Button("OK")]
    ]
    
    window_enter = sg.Window("Uzdevuma ievadīšana", layout = layout)
    
    try:
        while True:
            event, values = window_enter.read()
            
            if event == "OK" or event == sg.WIN_CLOSED:
                # pārbaudīt, lai ir skaitlis
                koefs = np.array(list(values.values()))
                if strikti:
                    index = funkc_nr * x_nr
                    # mērķa funkciju koeficienti
                    merka_koefs = koefs[:index].reshape(-1, x_nr)
                
                else:
                    # katrai piederības funkcijai ir 10 vērtības, kas jāsaglabā
                    index = funkc_nr_dal_2 * x_nr * 10
                    # mērķa funkciju koeficienti
                    merka_koefs = koefs[:index].reshape(-1, x_nr, 10)
                    
                    c_sympy = [[] for _ in range(x_nr)]
                    x = Symbol(f"x", real = True)
                    c_ij = np.zeros((2, funkc_nr_dal_2, x_nr))  
                ierob_index = x_nr + 2
    
                # ierobežojumu vērtības
                ierob_koefs = koefs[index:]
                
                if strikti:
                    # izveido mērķa funkcijas
                    for i in range(funkc_nr):
                        # izvēlas i-tās funkcijas koeficientus
                        koef = merka_koefs[i]
                        f_pulp[i] = sum(mn * float(kf) for mn, kf in zip(x_pulp, koef)) 
                        f_sympy[i] = sum(mn * float(kf) for mn, kf in zip(x_sympy, koef))
                
                else:
                    # izveido mērķa funkcijas
                    for i in range(funkc_nr_dal_2):
                        for j in range(x_nr):                
                            koef = merka_koefs[i,j]
                            koef = [parse_expr(k) for k in koef]
                            c_sympy[i].append(Piecewise((0, x <= koef[0]), 
                                                        (koef[1] * x + koef[2], ((koef[3] <= x) & (x <= koef[4]))), 
                                                        (koef[5] * x + koef[6], ((koef[7] <= x) & (x <= koef[8]))), 
                                                        (0, koef[9] <= x)))
                            c_ij[0,i,j], c_ij[1,i,j] = solve(c_sympy[i][j] - alfa)
    
    
                    for i in range(funkc_nr):
                        f_pulp[i] = sum(mn * float(kf) for mn, kf in zip(x_pulp, c_ij[int(i/funkc_nr_dal_2),i % 2]))
                        f_sympy[i] = sum(mn * float(kf) for mn, kf in zip(x_sympy, c_ij[int(i/funkc_nr_dal_2),i % 2]))
    
                    
                    
                # izveido ierobežojumus
                for i in range(ierob_nr):
                    # i-tā ierobežojuma koeficienti
                    koef = ierob_koefs[ierob_index*i:ierob_index*i+x_nr]
                    # i-tā ierobežojuma zīme
                    zime = ierob_koefs[ierob_index*i+x_nr]
                    # i-tā ierobežojum b vērtība
                    b = float(ierob_koefs[ierob_index*i+x_nr+1])
                    if zime == "=":
                        constraints.append(sum(mn * float(kf) for mn, kf in zip(x_pulp, koef)) == b)
                        constraints_sympy.append(Eq(sum(mn * float(kf) for mn, kf in zip(x_sympy, koef)), b))
                    elif zime == "<=":
                        constraints.append(sum(mn * float(kf) for mn, kf in zip(x_pulp, koef)) <= b)
                        constraints_sympy.append(LessThan(sum(mn * float(kf) for mn, kf in zip(x_sympy, koef)), b))
                    elif zime == ">=":
                        constraints.append(sum(mn * float(kf) for mn, kf in zip(x_pulp, koef)) >= b)
                        constraints_sympy.append(GreaterThan(sum(mn * float(kf) for mn, kf in zip(x_sympy, koef)), b))
                    
                break
            
    except Exception as error:
        window_enter.close()
        sg.popup_error_with_traceback(f'Radusies kļūda!', error)
    
    window_enter.close()
    
    return(f_pulp, f_sympy, constraints, constraints_sympy)

# ===================================================
# atrisina lpu_max problēmu (x_n >= 0)
# ===================================================
def lpu_max(obj, constraints):             # inputs ir ar pulp mainīgajiem
    lpu = pulp.LpProblem("LPU", pulp.LpMaximize)
    for constr in constraints:
        lpu += constr
    lpu += obj
    
    status = lpu.solve()
    if status != 1:
        return("Nav atrisināts")
    koord = {}
    for var in lpu.variables():
        koord[f"{var.name}"] = var.value()
    vert = lpu.objective.value()
    return (vert, koord)

# ===================================================
# atrod funkcijas minimumu no pieejamajiem maksimuma punktiem
# ===================================================
def find_min(f_sympy, f_max, x_sympy, kaarta, funkc_nr, strikti):
    if strikti:
        points_to_loop = [p for i, p in enumerate(f_max) if i != kaarta]
    else:
        points_to_loop = [p for i, p in enumerate(f_max) if (i + kaarta) % 2 != 0]
        
    sar = []
    for points in points_to_loop:
        vals = [val for val in points[1].values()]
        sar = sar + [[(x, val) for x, val in zip(x_sympy, vals)]]
        
    vert = []
    for punkts in sar:
        vert.append(f_sympy.subs(punkts))   
    return min(vert)

# ===================================================
# pārveido uzdevumu par LPU uzd. un atrisina
# ===================================================
def lpu_risinajums(f_pulp, constraints, f_min, f_max, funkc_nr):
    gamma = pulp.LpVariable(name="gamma")
    gamma_ierob = [gamma <= (f_pulp[i] - f_min[i]) / (f_max[i][0] - f_min[i]) for i in range(funkc_nr)]
    constraints += gamma_ierob
    max_pied, koord = lpu_max(gamma, constraints)
    koord.pop("gamma")
    return(max_pied, koord)

# ===================================================
# pārbauda vai koordinātas ir iekš definīcijas apgabala
# ===================================================
def check_constraints(koord, constraints, x_sympy):
    # pārliecinās, ka koordinātas ir iekš ierobežojumiem
    sub = as_sub(x_sympy, koord)
    for constr in constraints:
        if constr.subs(sub):
            continue
        else:
            return False
    return True

def as_sub(x_sympy, koord):
    return{x:y for x,y in zip(x_sympy, koord)}

# ===================================================
# izveido piederības funkciju
# ===================================================
def pied_f(f, f_min, f_max):
    return Piecewise( (0, f < f_min), 
                      (((f - f_min) / (f_max - f_min)), ((f_min <= f) & (f <= f_max))), 
                      (1, ((f_max < f))) ) 

# ===================================================
# skaitliski atrod funkcijas maksimumu
# ===================================================
def find_max(funkc, constraints_sympy, grad, x_sympy, epsilon, lr):
    koord = np.array([0 for x in range(len(x_sympy))])
    start = True
    val_old = 0
    init_val = 1
    
    
    layout = [
        [sg.Text("Risina...")],
        [sg.Text(" ", key = "progress")]
    ]
    
    window_risina = sg.Window("Risina", layout = layout, finalize = True)
    
    while True:
        window_risina.refresh()
        
        if init_val < 0.00001:
            sg.popup_ok("Nevar atrast sākuma punktu.")
            window_risina.close()
            break
    
        if start:
            val1 = funkc.subs(as_sub(x_sympy, koord))
            if round(val1, 10) == 0 or val1 == sympy.nan:
                if check_constraints(koord + init_val, constraints_sympy, x_sympy):
                    koord = koord + init_val
                else:
                    while not check_constraints(koord + init_val, constraints_sympy, x_sympy):
                        init_val *= 0.5
                    koord = koord + init_val
                continue
            else:
                start = False
        
        eval_grad = np.array([float(grad[i].subs(as_sub(x_sympy, koord))) for i in range(len(x_sympy))])
        koord_new = koord + lr * eval_grad

        while not check_constraints(koord_new, constraints_sympy, x_sympy):
            lr *= 0.5
            koord_new = koord + lr * eval_grad
        
        koord = koord_new
        val_new = funkc.subs(as_sub(x_sympy, koord))
        
        if abs((val_new - val_old) / val_old) < epsilon:
            window_risina.close()
            return (val_new, koord)
        val_old = val_new
        text = f"Piederība {val_new} \nKoordinātas {koord} \nLR {lr}"
        window_risina["progress"].update(value = text)
    
# ===================================================
# izveido minimumu un gradientu starp mi funkcijām (2+ funkcijas)
# ===================================================
def min_and_grad(mi, x_sympy):
    funkc = (mi[0] + mi[1] - abs(mi[0] - mi[1])) / 2
    if len(mi) > 2:
        for f in mi[2:]:
            funkc = (funkc + f - abs(funkc - f)) / 2
    grad = [funkc.diff(x) for x in x_sympy]
    return (funkc, grad)

# ===================================================
# izveido reizinājumu un gradientu starp mi funkcijām (2+ funkcijas)
# ===================================================
def prod_and_grad(mi, x_sympy):
    funkc = mi[0] * mi[1]
    if len(mi) > 2:
        for f in mi[2:]:
            funkc = funkc * f
    grad = [funkc.diff(x) for x in x_sympy]
    return (funkc, grad) 

# ===================================================
# izveido lukašēviča funkciju un gradientu starp mi funkcijām (2+ funkcijas)
# ===================================================
def luk_and_grad(mi, x_sympy):
    funkc = Max(mi[0] + mi[1] - 1, 0)
    if len(mi) > 2:
        for f in mi[2:]:
            funkc = Max(funkc + f - 1, 0)
    grad = [funkc.diff(x) for x in x_sympy]
    return (funkc, grad)

# ===================================================
# izveido hamačera funkciju un gradientu starp mi funkcijām (2+ funkcijas)
# ===================================================
def ham_and_grad(mi, x_sympy):
    funkc = Piecewise( (0, (mi[0] == 0) & (mi[1] == 0)),
                       ((mi[0] * mi[1]) / (mi[0] + mi[1] - mi[0] * mi[1]), (mi[0] != 0) & (mi[1] != 0)) )
    if len(mi) > 2:
        for f in mi[2:]:
            funkc = Piecewise( (0, (funkc == 0) & (f == 0)),
                               ((funkc * f) / (funkc + f - funkc * f), (funkc != 0) & (f != 0)) )
    grad = [funkc.diff(x) for x in x_sympy] 
    return (funkc, grad)

#%%
# ===================================================
# ===================================================
# Galvenā koda daļa
# ===================================================
# ===================================================


# ===================================================
# izvada logu un nolasa tā vērtības par uzdevumu
# ===================================================

veidi = ["Strikti skaitļi", "Nestrikti skaitļi"]

col1 = [[sg.Text("Ievadiet mainīgo skaitu:"), sg.Push(), 
     sg.In(size = (5,1), key = "mainigie")],
    [sg.Text("Ievadiet funkciju skaitu:"), sg.Push(), 
     sg.In(size = (5,1), key = "funkc")],
    [sg.Text("Ievadiet ierobežojumu skaitu:"), sg.Push(), 
     sg.In(size = (5,1), key = "ierob")],
    [sg.Button("OK")]]

col2 = [[sg.Button("Izmantot piemēru", key = "piemers", size = (10,5))]]

cols = [sg.Column(col1), sg.VerticalSeparator(), sg.Column(col2)]


layout = [[sg.Combo(veidi, default_value = veidi[0], readonly = True, key = "veids")], 
          [sg.Text("Alfa līmenis"), sg.In(default_text = "1", size = (6,1), key = "alfa")],
          [sg.HorizontalSeparator()],
          cols]

window_mainigie = sg.Window("Uzdevuma specifikācija", layout = layout)

try:
    while True:
        event, values = window_mainigie.read()
        
        if event == "OK":
            # pārbaudīt, lai ir skaitlis  
            x_nr = int(values["mainigie"])
            funkc_nr = int(values["funkc"])
            ierob_nr = int(values["ierob"])
            piemers = False
            strikti = True if values["veids"] == veidi[0] else False
            alfa = 1
            if not strikti:
                alfa = float(values["alfa"])
                if alfa == 1:
                    strikti = True
                else:
                    funkc_nr *= 2
            
            break            
        elif event == "piemers":
            x_nr = 2
            funkc_nr = 2
            ierob_nr = 4
            piemers = True
            strikti = True if values["veids"] == veidi[0] else False
            if not strikti:
                alfa = float(values["alfa"])
                if alfa == 1:
                    strikti = True
                else:
                    funkc_nr *= 2
                    
            break
        
        elif event == sg.WIN_CLOSED:
            break
        

except Exception as error:
    window_mainigie.close()
    sg.popup_error_with_traceback(f'Radusies kļūda!', error)



x_pulp = [pulp.LpVariable(f"x_{i+1}", lowBound = 0) for i in range(x_nr)]
x_sympy = [Symbol(f"x_{i+1}", real = True) for i in range(x_nr)]

f_pulp = [f"f_{i+1}" for i in range(funkc_nr)]
f_sympy = [f"f_{i+1}" for i in range(funkc_nr)]



#%%
# ===================================================
# ievada koeficientus vai izvēlas piemēru
# ===================================================
if piemers == False:
    window_mainigie.close()
    f_pulp, f_sympy, constraints, constraints_sympy = ievadit_uzd(x_nr, funkc_nr, ierob_nr, x_pulp, x_sympy, 
                                                                  f_pulp, f_sympy, strikti, alfa)
    
else:
    if strikti:
        constraints = []
        constraints_sympy = []
        f_pulp[0] = -1 * x_pulp[0] + 2 * x_pulp[1]
        f_pulp[1] = 2 * x_pulp[0] + 1 * x_pulp[1]
        f_sympy[0] = -1 * x_sympy[0] + 2 * x_sympy[1]
        f_sympy[1] = 2 * x_sympy[0] + 1 * x_sympy[1]
        constraints.append(-1 * x_pulp[0] + 3 * x_pulp[1] <= 21)
        constraints.append(1 * x_pulp[0] + 3 * x_pulp[1] <= 27)
        constraints.append(4 * x_pulp[0] + 3 * x_pulp[1] <= 45)
        constraints.append(3 * x_pulp[0] + 1 * x_pulp[1] <= 30)
        constraints_sympy.append(LessThan(-1 * x_sympy[0] + 3 * x_sympy[1], 21))
        constraints_sympy.append(LessThan(1 * x_sympy[0] + 3 * x_sympy[1], 27))
        constraints_sympy.append(LessThan(4 * x_sympy[0] + 3 * x_sympy[1], 45))
        constraints_sympy.append(LessThan(3 * x_sympy[0] + 1 * x_sympy[1], 30))
    

    else:
        constraints = []
        constraints_sympy = []
        x = Symbol("x", real = True)
        
        c_11 = Piecewise((0, x <= -2), 
                         (x + 2, ((-2 <= x) & (x <= -1))), 
                         (-x, ((-1 <= x) & (x <= 0))), 
                         (0, 0 <= x)) 
        c_12 = Piecewise((0, x <= 1), 
                         (x - 1, ((1 <= x) & (x <= 2))), 
                         (-x + 3, ((2 <= x) & (x <= 3))), 
                         (0, 3 <= x))  
        c_21 = Piecewise((0, x <= 1), 
                         (x - 1, ((1 <= x) & (x <= 2))), 
                         (-x + 3, ((2 <= x) & (x <= 3))), 
                         (0, 3 <= x))  
        c_22 = Piecewise((0, x <= 0), 
                         (x, ((0 <= x) & (x <= 1))), 
                         (2 - x, ((1 <= x) & (x <= 2))), 
                         (0, 2 <= x))
        
        c_11_k, c_11_l = solve(c_11 - alfa)
        c_12_k, c_12_l = solve(c_12 - alfa)
        c_21_k, c_21_l = solve(c_21 - alfa)
        c_22_k, c_22_l = solve(c_22 - alfa)
        
        f_pulp[0] = c_11_k * x_pulp[0] + c_12_k * x_pulp[1]
        f_pulp[1] = c_21_k * x_pulp[0] + c_22_k * x_pulp[1]
        f_pulp[2] = c_11_l * x_pulp[0] + c_12_l * x_pulp[1]
        f_pulp[3] = c_21_l * x_pulp[0] + c_22_l * x_pulp[1]
        
        f_sympy[0] = c_11_k * x_sympy[0] + c_12_k * x_sympy[1]
        f_sympy[1] = c_21_k * x_sympy[0] + c_22_k * x_sympy[1]
        f_sympy[2] = c_11_l * x_sympy[0] + c_12_l * x_sympy[1]
        f_sympy[3] = c_21_l * x_sympy[0] + c_22_l * x_sympy[1]
        
        constraints.append(-1 * x_pulp[0] + 3 * x_pulp[1] <= 21)
        constraints.append(1 * x_pulp[0] + 3 * x_pulp[1] <= 27)
        constraints.append(4 * x_pulp[0] + 3 * x_pulp[1] <= 45)
        constraints.append(3 * x_pulp[0] + 1 * x_pulp[1] <= 30)
        constraints_sympy.append(LessThan(-1 * x_sympy[0] + 3 * x_sympy[1], 21))
        constraints_sympy.append(LessThan(1 * x_sympy[0] + 3 * x_sympy[1], 27))
        constraints_sympy.append(LessThan(4 * x_sympy[0] + 3 * x_sympy[1], 45))
        constraints_sympy.append(LessThan(3 * x_sympy[0] + 1 * x_sympy[1], 30))
    
    window_mainigie.close()

f_max = [lpu_max(f, constraints) for f in f_pulp]
f_min = [find_min(f_sympy[i], f_max, x_sympy, i, funkc_nr, strikti) for i in range(funkc_nr)]


skait_metodes = ["minimuma", "reizinājuma", "lukašēviča", "hamačera"]


col1 = [[sg.Button("Risināt kā LPU", key = "lpu", size = (8,5))]]
col2 = [[sg.VerticalSeparator()]]
col3 = [[sg.Push(), sg.Text("Funkcija"), sg.Combo(skait_metodes, readonly = True, key = "metode")],
        [sg.Push(), sg.Text("Epsilons"), sg.In(default_text = "0.0001", size = (6,1), key = "epsilon")],
        [sg.Push(), sg.Text("LR"), sg.In(default_text = "2", size = (6,1), key = "lr")],
        [sg.Push(), sg.Button("Risināt skaitliski", key = "skait")]]

layout = [[sg.Column(col1), sg.VerticalSeparator(), sg.Column(col3)]]

window_veids = sg.Window("Risināšanas veids", layout = layout)

# ===================================================
# risina uzdevumu izvēlētajā veidā
# ===================================================
try:
    while True:
        event, values = window_veids.read()
        
        if event == "lpu":
            max_pied, koord = lpu_risinajums(f_pulp, constraints, f_min, f_max, funkc_nr)
            break
        elif event == "skait":
            epsilon = float(values["epsilon"])
            lr = float(values["lr"])
            
            mi = [pied_f(f_sympy[i], f_min[i], f_max[i][0]) for i in range(funkc_nr)]
            if values["metode"] == skait_metodes[0]:
                funkc, grad = min_and_grad(mi, x_sympy)
                window_veids.close()
                max_pied, koord = find_max(funkc, constraints_sympy, grad, x_sympy, epsilon, lr)
               
                break
            elif values["metode"] == skait_metodes[1]:
                funkc, grad = prod_and_grad(mi, x_sympy)
                window_veids.close()
                max_pied, koord = find_max(funkc, constraints_sympy, grad, x_sympy, epsilon, lr)
                
                break
            elif values["metode"] == skait_metodes[2]:
                funkc, grad = luk_and_grad(mi, x_sympy)
                window_veids.close()
                max_pied, koord = find_max(funkc, constraints_sympy, grad, x_sympy, epsilon, lr)
                
                break
            elif values["metode"] == skait_metodes[3]:
                # strādās tikai ar 2 funkcijām
                funkc, grad = ham_and_grad(mi, x_sympy)
                window_veids.close()
                max_pied, koord = find_max(funkc, constraints_sympy, grad, x_sympy, epsilon, lr)
                
                break
        elif event == sg.WIN_CLOSED:
            break
        
except Exception as error:
    window_veids.close()
    sg.popup_error_with_traceback(f'Radusies kļūda!', error)
        
window_veids.close()


#%%
# ===================================================
# rezultāta logs
# ===================================================
layout = [
    [sg.Text(f"Risinājums ar piederību {max_pied} atrodas punktā {koord}.")],
    [sg.Button("OK")]
]

window_rezultats = sg.Window("Rezultāts", layout = layout)

while True:
    event, values = window_rezultats.read()
    
    if event == "OK" or event == sg.WIN_CLOSED:      
        break
    
window_rezultats.close()