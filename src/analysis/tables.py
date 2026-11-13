"""Write self-contained LaTeX tabular results."""
import math


def _cell(value):
    if value is None:return '--'
    if isinstance(value, (int, float)):
        if not math.isfinite(value):return '--'
        return f'{value:.3f}' if isinstance(value,float) else str(value)
    substitutions={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','<':r'\ensuremath{<}','>':r'\ensuremath{>}'}
    return ''.join(substitutions.get(char,char) for char in str(value))


def write_latex(path,headers,rows):
    lines=[r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\hline', ' & '.join(map(_cell,headers))+r' \\',r'\hline']
    for row in rows:lines.append(' & '.join(_cell(value) for value in row)+r' \\')
    lines.extend([r'\hline',r'\end{tabular}'])
    path.write_text('\n'.join(lines)+'\n')
