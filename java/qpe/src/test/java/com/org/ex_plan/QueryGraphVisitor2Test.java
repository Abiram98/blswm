package com.org.ex_plan;
import static org.junit.Assert.assertTrue;

import org.apache.jena.base.Sys;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import java.util.HashMap;
import java.util.ArrayList;
import org.junit.Test;

public class QueryGraphVisitor2Test {
    @Test
    public void shouldAnswerWithTrue()
    {
        assertTrue( true );
    }


    @Test
    public void graphConstructionTest(){
        String query = """
        PREFIX wdt: <http://www.wikidata.org/entity/>
        SELECT ?pres ?occName ?presName WHERE {
            ?pres wdt:occupation ?occ .
            ?pres wdt:position/wdt:subclassOf* wdt:President .
            ?occ wdt:label ?occName .
            OPTIONAL {
                ?pres wdt:label ?presName .
            }
            FILTER (!REGEX(?occName, ".*[Pp]resident.*"))
                }""";
        String graph = (String) QueryGraphVisitor2.QGCreate(query);
        System.out.println(graph);
        
    }
}
