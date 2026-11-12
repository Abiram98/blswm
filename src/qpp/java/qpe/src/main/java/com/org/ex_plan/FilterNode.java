package com.org.ex_plan;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public class FilterNode extends Node{
    public String expression;
    public String nodeType;
    public ArrayList<String> vars = new ArrayList<>();
    public Set<String> operationTypes = new HashSet<>();

    public FilterNode(String expression) {
        this.expression = expression;
        nodeType = "FILTER";
    }

    @Override
    public String getJoinedVar(String term) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unsupported method 'getJoinedVar'");
    }

    @Override
    public ArrayList<String> getTerms() {
        return vars;
    }

    public void addVars(Collection<String> varNames){
        vars.addAll(varNames);
    }
    
    public void addOperationTypes(Set<String> opTypes){
        operationTypes.addAll(opTypes);
    }
    
    public String toString(){
        return this.nodeType + ": " + this.expression + " Vars: " + this.vars.toString() + " Operations: " + this.operationTypes.toString();
    }
}
