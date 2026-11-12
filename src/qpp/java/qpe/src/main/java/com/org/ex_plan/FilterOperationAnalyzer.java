package com.org.ex_plan;

import java.util.HashSet;
import java.util.Set;

import org.apache.jena.sparql.expr.Expr;
import org.apache.jena.sparql.expr.ExprAggregator;
import org.apache.jena.sparql.expr.ExprFunction;
import org.apache.jena.sparql.expr.ExprFunction0;
import org.apache.jena.sparql.expr.ExprFunction1;
import org.apache.jena.sparql.expr.ExprFunction2;
import org.apache.jena.sparql.expr.ExprFunction3;
import org.apache.jena.sparql.expr.ExprFunctionN;
import org.apache.jena.sparql.expr.ExprFunctionOp;
import org.apache.jena.sparql.expr.ExprNone;
import org.apache.jena.sparql.expr.ExprTripleTerm;
import org.apache.jena.sparql.expr.ExprVar;
import org.apache.jena.sparql.expr.ExprVisitor;
import org.apache.jena.sparql.expr.NodeValue;

/**
 * Analyzer to traverse ARQ filter expressions and categorize operation types.
 * 
 * Operation categories:
 * 1. Arithmetic: +, *, /, -
 * 2. Logical: ||, &&
 * 3. Comparison: =, !=, <, >, >=, <=
 * 4. General: DATATYPE, STR, IRI, LANG, BOUND, IN, NOT IN, isBlank, isIRI, isLiteral
 * 5. String: STRLEN, SUBSTR, UCASE, LCASE, STRSTARTS, STRENDS, CONTAINS, STRBEFORE, STRAFTER, 
 *           ENCODE_FOR_URI, CONCAT, LANGMATCHES, REGEX, REPLACE
 * 6. Time: NOW, YEAR, MONTH, DAY, HOUR, MINUTES, SECONDS, TIMEZONE, TZ
 */
public class FilterOperationAnalyzer implements ExprVisitor {
    
    public Set<String> operationTypes = new HashSet<>();
    
    // Operator symbol mappings
    private static final Set<String> ARITHMETIC_OPS = Set.of("+", "-", "*", "/");
    private static final Set<String> LOGICAL_OPS = Set.of("||", "&&", "!");
    private static final Set<String> COMPARISON_OPS = Set.of("=", "!=", "<", ">", "<=", ">=");
    
    // Function name mappings
    private static final Set<String> GENERAL_FUNCTIONS = Set.of(
        "datatype", "str", "iri", "uri", "lang", "bound", "in", "notin", 
        "isblank", "isiri", "isuri", "isliteral"
    );
    
    private static final Set<String> STRING_FUNCTIONS = Set.of(
        "strlen", "substr", "ucase", "lcase", "strstarts", "strends", 
        "contains", "strbefore", "strafter", "encode_for_uri", "concat", 
        "langmatches", "regex", "replace"
    );
    
    private static final Set<String> TIME_FUNCTIONS = Set.of(
        "now", "year", "month", "day", "hours", "minutes", "seconds", 
        "timezone", "tz"
    );
    
    public FilterOperationAnalyzer() {
    }
    
    /**
     * Analyze an expression and populate operation types
     */
    public void analyze(Expr expr) {
        if (expr != null) {
            expr.visit(this);
        }
    }
    
    /**
     * Check if an operator symbol is arithmetic
     */
    private void checkArithmetic(String symbol) {
        if (ARITHMETIC_OPS.contains(symbol)) {
            operationTypes.add("ARITHMETIC");
        }
    }
    
    /**
     * Check if an operator symbol is logical
     */
    private void checkLogical(String symbol) {
        if (LOGICAL_OPS.contains(symbol)) {
            operationTypes.add("LOGICAL");
        }
    }
    
    /**
     * Check if an operator symbol is comparison
     */
    private void checkComparison(String symbol) {
        if (COMPARISON_OPS.contains(symbol)) {
            operationTypes.add("COMPARISON");
        }
    }
    
    /**
     * Categorize a function by name
     */
    private void categorizeFunction(ExprFunction func) {
        String funcName = func.getFunctionSymbol().getSymbol().toLowerCase();
        
        // Remove namespace prefix if present
        if (funcName.contains("#")) {
            funcName = funcName.substring(funcName.indexOf("#") + 1);
        }
        if (funcName.contains(":")) {
            funcName = funcName.substring(funcName.indexOf(":") + 1);
        }
        
        if (GENERAL_FUNCTIONS.contains(funcName)) {
            operationTypes.add("GENERAL");
        } else if (STRING_FUNCTIONS.contains(funcName)) {
            operationTypes.add("STRING");
        } else if (TIME_FUNCTIONS.contains(funcName)) {
            operationTypes.add("TIME");
        }
    }
    
    @Override
    public void visit(ExprFunction0 func) {
        categorizeFunction(func);
    }
    
    @Override
    public void visit(ExprFunction1 func) {
        categorizeFunction(func);
        
        // Check operator symbol for arithmetic/logical/comparison
        String opName = func.getOpName();
        if (opName != null) {
            checkArithmetic(opName);
            checkLogical(opName);
            checkComparison(opName);
        }
        
        // Recurse into argument
        if (func.getArg() != null) {
            func.getArg().visit(this);
        }
    }
    
    @Override
    public void visit(ExprFunction2 func) {
        categorizeFunction(func);
        
        // Check operator symbol for arithmetic/logical/comparison
        String opName = func.getOpName();
        if (opName != null) {
            checkArithmetic(opName);
            checkLogical(opName);
            checkComparison(opName);
        }
        
        // Recurse into arguments
        if (func.getArg1() != null) {
            func.getArg1().visit(this);
        }
        if (func.getArg2() != null) {
            func.getArg2().visit(this);
        }
    }
    
    @Override
    public void visit(ExprFunction3 func) {
        categorizeFunction(func);
        
        // Recurse into arguments
        if (func.getArg1() != null) {
            func.getArg1().visit(this);
        }
        if (func.getArg2() != null) {
            func.getArg2().visit(this);
        }
        if (func.getArg3() != null) {
            func.getArg3().visit(this);
        }
    }
    
    @Override
    public void visit(ExprFunctionN func) {
        categorizeFunction(func);
        
        // Recurse into all arguments
        for (Expr arg : func.getArgs()) {
            if (arg != null) {
                arg.visit(this);
            }
        }
    }
    
    @Override
    public void visit(ExprFunctionOp funcOp) {
        // Recurse into the element if present
        if (funcOp.getElement() != null && funcOp.getGraphPattern() != null) {
            // For EXISTS/NOT EXISTS patterns
            operationTypes.add("LOGICAL");
        }
    }
    
    @Override
    public void visit(NodeValue nv) {
        // Leaf node - no action needed
    }
    
    @Override
    public void visit(ExprVar nv) {
        // Variable - no action needed
    }
    
    @Override
    public void visit(ExprAggregator eAgg) {
        // Aggregator - recurse into the expression if it has one
        if (eAgg.getAggregator() != null) {
            // Aggregators typically involve functions, mark as general for now
            operationTypes.add("GENERAL");
        }
    }
    
    @Override
    public void visit(ExprNone none) {
        // No expression - no action needed
    }
    
    @Override
    public void visit(ExprTripleTerm tripleTerm) {
        // Triple term - no specific operation type
    }
}
