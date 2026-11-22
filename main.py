import tkinter as tk
from tkinter import ttk, messagebox
import math
import ast
import sympy as sp

# ---------------- Safe mathematical expression handling ----------------
_ALLOWED_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "log10": math.log10, "abs": abs, "pi": math.pi, "e": math.e
}

def compile_expr(expr):
    expr = expr.strip().replace("^", "**")
    if not expr:
        raise ValueError("Please enter a function.")
    tree = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
        ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.USub, ast.UAdd, ast.Call
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("Unsupported expression. Use numbers, x, + - * / ^ and math functions.")
        if isinstance(node, ast.Name) and node.id not in set(_ALLOWED_FUNCS) | {"x"}:
            raise ValueError(f"Unknown name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("Only supported math functions can be called.")
    code = compile(tree, "<function>", "eval")
    def f(x):
        return float(eval(code, {"__builtins__": {}}, {**_ALLOWED_FUNCS, "x": float(x)}))
    return f, expr

def derivative_expr(expr):
    x = sp.symbols("x")
    safe = {k: getattr(sp, k) for k in ("sin","cos","tan","asin","acos","atan","sinh","cosh","tanh","sqrt","exp","log","Abs")}
    safe.update({"pi": sp.pi, "e": sp.E})
    try:
        symbolic = sp.sympify(expr.replace("^", "**"), locals=safe)
        deriv = sp.diff(symbolic, x)
        fn = sp.lambdify(x, deriv, modules=["math"])
        return lambda value: float(fn(value)), str(deriv)
    except Exception:
        # Numerical central difference fallback
        f, _ = compile_expr(expr)
        return lambda value: (f(value + 1e-6) - f(value - 1e-6)) / 2e-6, "numerical derivative"

# ---------------- Numerical algorithms ----------------
def bisection(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    if fa == 0: return a, 0, [["0", a, fa, b, fb, a, 0, 0]]
    if fb == 0: return b, 0, [["0", a, fa, b, fb, b, 0, 0]]
    if fa * fb > 0:
        raise ValueError("Bisection requires f(a) and f(b) to have opposite signs.")
    rows = []
    c_old = None
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        err = abs(c - c_old) if c_old is not None else abs(b-a)/2
        rows.append([i, a, fa, b, fb, c, fc, err])
        if abs(fc) < tol or err < tol:
            return c, i, rows
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        c_old = c
    return c, max_iter, rows

def false_position(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("False Position requires f(a) and f(b) to have opposite signs.")
    rows, old = [], None
    for i in range(1, max_iter + 1):
        den = fb - fa
        if abs(den) < 1e-15: raise ValueError("Division by zero in False Position.")
        c = (a*fb - b*fa) / den
        fc = f(c)
        err = abs(c-old) if old is not None else abs(b-a)
        rows.append([i, a, fa, b, fb, c, fc, err])
        if abs(fc) < tol or err < tol: return c, i, rows
        if fa * fc < 0: b, fb = c, fc
        else: a, fa = c, fc
        old = c
    return c, max_iter, rows

def newton(f, df, x, tol, max_iter):
    rows = []
    for i in range(1, max_iter + 1):
        fx, dfx = f(x), df(x)
        if abs(dfx) < 1e-14: raise ValueError("Derivative is zero or too small.")
        xn = x - fx/dfx
        err = abs(xn-x)
        rows.append([i, x, fx, dfx, xn, f(xn), err])
        x = xn
        if err < tol or abs(f(x)) < tol: return x, i, rows
    return x, max_iter, rows

def secant(f, x0, x1, tol, max_iter):
    rows = []
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        den = f1-f0
        if abs(den) < 1e-14: raise ValueError("Secant denominator is zero.")
        x2 = x1 - f1*(x1-x0)/den
        err = abs(x2-x1)
        rows.append([i, x0, f0, x1, f1, x2, f(x2), err])
        x0, x1 = x1, x2
        if err < tol or abs(f(x1)) < tol: return x1, i, rows
    return x1, max_iter, rows

def fixed_point(g, x, tol, max_iter, f=None):
    rows = []
    for i in range(1, max_iter + 1):
        xn = g(x)
        err = abs(xn-x)
        residual = f(xn) if f else float("nan")
        rows.append([i, x, xn, residual, err])
        x = xn
        if err < tol: return x, i, rows
    return x, max_iter, rows

def solve_linear(A, b, method, tol, max_iter, x0):
    n = len(A)
    M = [list(map(float, row)) for row in A]
    rhs = list(map(float, b))
    rows = []
    if method == "Doolittle":
        L = [[0.0]*n for _ in range(n)]
        U = [[0.0]*n for _ in range(n)]
        for i in range(n): L[i][i] = 1.0
        for j in range(n):
            for i in range(j+1):
                U[i][j] = M[i][j] - sum(L[i][k]*U[k][j] for k in range(i))
            if abs(U[j][j]) < 1e-14: raise ValueError("Zero pivot in Doolittle; try a different equation order.")
            for i in range(j+1,n):
                L[i][j] = (M[i][j]-sum(L[i][k]*U[k][j] for k in range(j)))/U[j][j]
            rows.append([f"Factorization step {j+1}", [r[:] for r in L], [r[:] for r in U]])
        y = [0.0]*n
        for i in range(n): y[i] = rhs[i]-sum(L[i][j]*y[j] for j in range(i))
        x = [0.0]*n
        for i in range(n-1,-1,-1): x[i]=(y[i]-sum(U[i][j]*x[j] for j in range(i+1,n)))/U[i][i]
        return x, len(rows), rows

    if method == "Thomas":
        # Thomas algorithm for tridiagonal systems
        for i in range(n):
            for j in range(n):
                if abs(i-j)>1 and abs(M[i][j])>1e-12:
                    raise ValueError("Thomas method requires a tridiagonal coefficient matrix.")
        c = [M[i][i+1] if i<n-1 else 0.0 for i in range(n)]
        d = rhs[:]
        a = [M[i][i-1] if i>0 else 0.0 for i in range(n)]
        if abs(M[0][0]) < 1e-14: raise ValueError("Zero pivot in Thomas algorithm.")
        for i in range(1,n):
            w = a[i]/M[i-1][i-1]
            M[i][i] -= w*c[i-1]
            d[i] -= w*d[i-1]
            rows.append([f"Forward sweep {i}", M[i][i], d[i]])
            if abs(M[i][i]) < 1e-14: raise ValueError("Zero pivot in Thomas algorithm.")
        x=[0.0]*n; x[-1]=d[-1]/M[-1][-1]
        for i in range(n-2,-1,-1): x[i]=(d[i]-c[i]*x[i+1])/M[i][i]
        return x, len(rows), rows

    if method == "Jacobi":
        x = list(x0) if x0 else [0.0]*n
        if len(x) != n: raise ValueError("Enter one initial value for each unknown.")
        for i in range(n):
            if abs(M[i][i]) < 1e-14: raise ValueError("A diagonal coefficient is zero.")
        for it in range(1,max_iter+1):
            xn = []
            for i in range(n):
                xn.append((rhs[i]-sum(M[i][j]*x[j] for j in range(n) if j!=i))/M[i][i])
            err = max(abs(xn[i]-x[i]) for i in range(n))
            rows.append([it, x[:], xn[:], err])
            x = xn
            if err < tol: return x, it, rows
        return x, max_iter, rows

    if method == "Gauss-Seidel":
        x = list(x0) if x0 else [0.0]*n
        if len(x) != n: raise ValueError("Enter one initial value for each unknown.")
        for i in range(n):
            if abs(M[i][i]) < 1e-14: raise ValueError("A diagonal coefficient is zero.")
        for it in range(1, max_iter + 1):
            old = x[:]
            for i in range(n):
                x[i] = (rhs[i] - sum(M[i][j] * x[j] for j in range(n) if j != i)) / M[i][i]
            err = max(abs(x[i] - old[i]) for i in range(n))
            rows.append([it, old, x[:], err])
            if err < tol:
                return x, it, rows
        return x, max_iter, rows
    raise ValueError("Unknown method.")

# ---------------- GUI ----------------
BG = "#10243b"; PANEL = "#183650"; INPUT = "#24445f"
ACCENT = "#20b9d2"; TEXT = "#f4f7fb"; MUTED = "#b8cad8"

class SolverApp:
    def __init__(self, root):
        self.root = root
        root.title("Numerical Methods Solver")
        root.geometry("1180x850")
        root.minsize(950,700)
        root.configure(bg=BG)
        self.entries = {}
        self.matrix_entries = []
        self.rhs_entries = []
        self.initial_entries = []
        self.build_style()
        self.build_ui()
        self.on_method_change()

    def build_style(self):
        style=ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=INPUT, background=INPUT, foreground=TEXT, arrowcolor=TEXT)
        style.configure("Treeview", background="#102b43", fieldbackground="#102b43", foreground=TEXT, rowheight=27)
        style.configure("Treeview.Heading", background="#24516d", foreground=TEXT, font=("Segoe UI",10,"bold"))
        style.map("Treeview", background=[("selected","#147f9c")])

    def label(self,parent,text):
        return tk.Label(parent,text=text,bg=parent.cget("bg"),fg=TEXT,font=("Segoe UI",10,"bold"))

    def entry(self,parent,key,width=18,default=""):
        e=tk.Entry(parent,width=width,bg=INPUT,fg=TEXT,insertbackground="white",relief="flat",
                   font=("Consolas",11),highlightthickness=1,highlightbackground="#4e7893")
        e.insert(0,str(default)); self.entries[key]=e
        return e

    def build_ui(self):
        header=tk.Frame(self.root,bg="#0c1b2d",height=110); header.pack(fill="x")
        tk.Label(header,text="NUMERICAL METHODS",bg="#0c1b2d",fg="white",font=("Segoe UI",25,"bold")).pack(anchor="w",padx=28,pady=(17,0))
        tk.Label(header,text="Interactive numerical calculator  •  Python Desktop GUI",bg="#0c1b2d",fg="#c7d8e5",font=("Segoe UI",11)).pack(anchor="w",padx=30,pady=(0,15))
        top=tk.Frame(self.root,bg=PANEL,padx=18,pady=12); top.pack(fill="x",padx=18,pady=(15,8))
        self.label(top,"Choose method").grid(row=0,column=0,padx=(0,12),sticky="w")
        self.method=ttk.Combobox(top,state="readonly",width=25,values=[
            "Doolittle","Thomas","Jacobi","Gauss-Seidel","Bisection",
            "False Position","Newton-Raphson","Secant","Fixed Point"])
        self.method.grid(row=0,column=1,padx=8); self.method.bind("<<ComboboxSelected>>",lambda e:self.on_method_change())
        self.label(top,"Tolerance").grid(row=0,column=2,padx=(24,6))
        self.entry(top,"tol",12,"0.0001").grid(row=0,column=3,padx=5)
        self.label(top,"Max iterations").grid(row=0,column=4,padx=(20,6))
        self.entry(top,"max_iter",9,"100").grid(row=0,column=5,padx=5)
        self.input_panel=tk.Frame(self.root,bg=PANEL,padx=18,pady=14)
        self.input_panel.pack(fill="x",padx=18,pady=8)
        buttons=tk.Frame(self.root,bg=BG); buttons.pack(fill="x",padx=18,pady=8)
        tk.Button(buttons,text="CALCULATE  ▶",command=self.calculate,bg=ACCENT,fg="white",font=("Segoe UI",11,"bold"),relief="flat",padx=18,pady=9).pack(side="left")
        tk.Button(buttons,text="Clear results",command=self.clear_results,bg="#e8edf2",fg="#23384b",font=("Segoe UI",10),relief="flat",padx=16,pady=9).pack(side="left",padx=10)
        tk.Label(buttons,text="Angles are in radians  •  Use ^ for powers",bg=BG,fg=MUTED,font=("Segoe UI",9)).pack(side="right")
        tk.Label(self.root,text="Results",bg=BG,fg="white",font=("Segoe UI",20,"bold")).pack(anchor="w",padx=22,pady=(8,4))
        self.status=tk.Label(self.root,text="Choose a method and enter the required values.",bg="#24516d",fg="#d6f8ff",anchor="w",padx=12,pady=10,font=("Segoe UI",10))
        self.status.pack(fill="x",padx=18,pady=(0,8))
        self.summary=tk.Label(self.root,text="",bg=BG,fg="white",font=("Segoe UI",17,"bold"),anchor="w")
        self.summary.pack(fill="x",padx=22,pady=4)
        tableframe=tk.Frame(self.root,bg=BG); tableframe.pack(fill="both",expand=True,padx=18,pady=(4,16))
        self.tree=ttk.Treeview(tableframe,show="headings")
        vs=ttk.Scrollbar(tableframe,orient="vertical",command=self.tree.yview); hs=ttk.Scrollbar(tableframe,orient="horizontal",command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set,xscrollcommand=hs.set)
        self.tree.grid(row=0,column=0,sticky="nsew"); vs.grid(row=0,column=1,sticky="ns"); hs.grid(row=1,column=0,sticky="ew")
        tableframe.rowconfigure(0,weight=1); tableframe.columnconfigure(0,weight=1)

    def clear_panel(self):
        # Keep the global controls: calculate() always reads tolerance and max_iter.
        keep = {key: self.entries[key] for key in ("tol", "max_iter") if key in self.entries}
        for widget in self.input_panel.winfo_children():
            widget.destroy()
        self.entries = keep
        self.matrix_entries = []
        self.rhs_entries = []
        self.initial_entries = []

    def add_field(self,parent,label,key,row,col=0,default="",width=20):
        self.label(parent,label).grid(row=row,column=col,sticky="w",padx=6,pady=7)
        e=self.entry(parent,key,width,default); e.grid(row=row,column=col+1,sticky="w",padx=6,pady=7)
        return e

    def on_method_change(self):
        self.clear_panel()
        m=self.method.get()
        if not m: return
        if m in ("Bisection","False Position"):
            self.add_field(self.input_panel,"Function f(x)","func",0,0,"x^3-x-2",45)
            self.add_field(self.input_panel,"Lower bound a","a",1,0,"1")
            self.add_field(self.input_panel,"Upper bound b","b",1,2,"2")
        elif m=="Newton-Raphson":
            self.add_field(self.input_panel,"Function f(x)","func",0,0,"x^4-11*x+8",45)
            self.add_field(self.input_panel,"Initial x0","x0",1,0,"2")
            tk.Label(self.input_panel,text="Derivative is calculated automatically.",bg=PANEL,fg=MUTED).grid(row=1,column=2,sticky="w",padx=10)
        elif m=="Secant":
            self.add_field(self.input_panel,"Function f(x)","func",0,0,"x^3-x-2",45)
            self.add_field(self.input_panel,"Initial x0","x0",1,0,"1")
            self.add_field(self.input_panel,"Second guess x1","x1",1,2,"2")
        elif m=="Fixed Point":
            self.add_field(self.input_panel,"g(x) for Fixed Point","g",0,0,"(x+2)**(1/3)",35)
            self.add_field(self.input_panel,"Initial x0","x0",0,2,"1.5")
            self.add_field(self.input_panel,"f(x) (optional residual)","func",1,0,"x^3-x-2",35)
        else:
            bar=tk.Frame(self.input_panel,bg=PANEL); bar.grid(row=0,column=0,columnspan=12,sticky="w")
            self.label(bar,"Number of unknowns").pack(side="left",padx=5)
            self.nvar=tk.StringVar(value="3")
            self.ncombo=ttk.Combobox(bar,textvariable=self.nvar,values=[str(i) for i in range(2,9)],width=5,state="readonly")
            self.ncombo.pack(side="left",padx=8); self.ncombo.bind("<<ComboboxSelected>>",lambda e:self.build_matrix())
            self.label(bar,"Enter coefficient matrix A and constants b").pack(side="left",padx=18)
            self.matrix_frame=tk.Frame(self.input_panel,bg=PANEL); self.matrix_frame.grid(row=1,column=0,columnspan=12,sticky="w",pady=8)
            self.build_matrix()

    def build_matrix(self):
        for w in self.matrix_frame.winfo_children(): w.destroy()
        self.matrix_entries=[]; self.rhs_entries=[]; self.initial_entries=[]
        n=int(self.nvar.get())
        tk.Label(self.matrix_frame,text="A",bg=PANEL,fg=ACCENT,font=("Segoe UI",11,"bold")).grid(row=0,column=0,padx=5)
        for j in range(n): tk.Label(self.matrix_frame,text=f"x{j+1}",bg=PANEL,fg=TEXT).grid(row=0,column=j+1,padx=4)
        tk.Label(self.matrix_frame,text="b",bg=PANEL,fg=TEXT).grid(row=0,column=n+1,padx=8)
        tk.Label(self.matrix_frame,text="Initial value (Jacobi / Gauss-Seidel)",bg=PANEL,fg=TEXT).grid(row=0,column=n+2,padx=8)
        defaults=[[10,-1,2],[ -1,11,-1],[2,-1,10]]
        for i in range(n):
            row=[]
            tk.Label(self.matrix_frame,text=f"Eq. {i+1}",bg=PANEL,fg=MUTED).grid(row=i+1,column=0,padx=5,pady=3)
            for j in range(n):
                val=defaults[i][j] if n==3 else (1 if i==j else 0)
                e=tk.Entry(self.matrix_frame,width=8,bg=INPUT,fg=TEXT,insertbackground="white",relief="flat")
                e.insert(0,str(val)); e.grid(row=i+1,column=j+1,padx=3,pady=3); row.append(e)
            self.matrix_entries.append(row)
            be=tk.Entry(self.matrix_frame,width=8,bg=INPUT,fg=TEXT,insertbackground="white",relief="flat")
            be.insert(0,str([6,25,-11][i] if n==3 else 1)); be.grid(row=i+1,column=n+1,padx=5); self.rhs_entries.append(be)
            ie=tk.Entry(self.matrix_frame,width=8,bg=INPUT,fg=TEXT,insertbackground="white",relief="flat")
            ie.insert(0,"0"); ie.grid(row=i+1,column=n+2,padx=5); self.initial_entries.append(ie)

    def get_float(self,key):
        return float(self.entries[key].get().strip())

    def calculate(self):
        try:
            method=self.method.get()
            tol=self.get_float("tol"); max_iter=int(self.entries["max_iter"].get())
            if tol<=0 or max_iter<1: raise ValueError("Tolerance and maximum iterations must be positive.")
            if method in ("Bisection","False Position","Newton-Raphson","Secant","Fixed Point"):
                f=None
                if "func" in self.entries and self.entries["func"].get().strip():
                    f,_=compile_expr(self.entries["func"].get())
                if method in ("Bisection","False Position"):
                    if f is None: raise ValueError("Enter f(x).")
                    a=self.get_float("a"); b=self.get_float("b")
                    root,it,rows=(bisection if method=="Bisection" else false_position)(f,a,b,tol,max_iter)
                    cols=["Iteration","a","f(a)","b","f(b)","x","f(x)","Error"]
                elif method=="Newton-Raphson":
                    if f is None: raise ValueError("Enter f(x).")
                    df,dexpr=derivative_expr(self.entries["func"].get())
                    root,it,rows=newton(f,df,self.get_float("x0"),tol,max_iter)
                    cols=["Iteration","x(n)","f(xn)","f'(xn)","x(n+1)","f(xn+1)","Error"]
                    self.status.config(text=f"Newton-Raphson completed. Derivative used: {dexpr}")
                elif method=="Secant":
                    if f is None: raise ValueError("Enter f(x).")
                    root,it,rows=secant(f,self.get_float("x0"),self.get_float("x1"),tol,max_iter)
                    cols=["Iteration","x0","f(x0)","x1","f(x1)","x2","f(x2)","Error"]
                else:
                    g,_=compile_expr(self.entries["g"].get())
                    root,it,rows=fixed_point(g,self.get_float("x0"),tol,max_iter,f)
                    cols=["Iteration","x(n)","x(n+1)","f(xn+1)","Error"]
            else:
                A=[[float(e.get()) for e in row] for row in self.matrix_entries]
                b=[float(e.get()) for e in self.rhs_entries]
                x0=[float(e.get()) for e in self.initial_entries]
                root,it,rows=solve_linear(A,b,method,tol,max_iter,x0)
                if method in ("Jacobi", "Gauss-Seidel"):
                    cols=["Iteration","Old values","New values (updated in-place)","Max error"]
                elif method=="Doolittle": cols=["Step","L matrix","U matrix"]
                elif method=="Thomas": cols=["Step","Updated diagonal","Updated RHS"]
            self.show_table(cols,rows)
            if isinstance(root,list):
                summary="   ".join(f"x{i+1} = {v:.10g}" for i,v in enumerate(root))
                self.summary.config(text=f"Solution:  {summary}    |    Iterations/steps: {it}")
            else:
                self.summary.config(text=f"Final root:  x ≈ {root:.12g}    |    Iterations: {it}")
            if not self.status.cget("text").startswith("Newton-Raphson completed"):
                self.status.config(text=f"{method} completed successfully.")
        except Exception as ex:
            self.summary.config(text="")
            self.status.config(text=f"Could not calculate: {ex}")
            messagebox.showerror("Input / Calculation Error",str(ex))

    def show_table(self,cols,rows):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"]=list(range(len(cols)))
        for i,c in enumerate(cols):
            self.tree.heading(i,text=c); self.tree.column(i,width=max(110,min(260,len(c)*12+45)),anchor="center")
        for row in rows:
            vals=[]
            for v in row:
                if isinstance(v,float): vals.append(f"{v:.10g}")
                elif isinstance(v,list):
                    if v and isinstance(v[0],list):
                        vals.append(" ; ".join("["+", ".join(f"{z:.5g}" for z in r)+"]" for r in v))
                    else: vals.append("["+", ".join(f"{z:.7g}" if isinstance(z,(int,float)) else str(z) for z in v)+"]")
                else: vals.append(str(v))
            self.tree.insert("","end",values=vals)

    def clear_results(self):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"]=[]
        self.summary.config(text="")
        self.status.config(text="Results cleared. Enter values and calculate again.")

if __name__=="__main__":
    root=tk.Tk()
    app=SolverApp(root)
    root.mainloop()
