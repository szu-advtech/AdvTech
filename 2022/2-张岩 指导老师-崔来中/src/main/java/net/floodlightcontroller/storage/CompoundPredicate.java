package net.floodlightcontroller.storage;
public class CompoundPredicate implements IPredicate {
    public enum Operator { AND, OR };
    private Operator operator;
    private boolean negated;
    private IPredicate[] predicateList;
    public CompoundPredicate(Operator operator, boolean negated, IPredicate... predicateList) {
        this.operator = operator;
        this.negated = negated;
        this.predicateList = predicateList;
    }
    public Operator getOperator() {
        return operator;
    }
    public boolean isNegated() {
        return negated;
    }
    public IPredicate[] getPredicateList() {
        return predicateList;
    }
}
