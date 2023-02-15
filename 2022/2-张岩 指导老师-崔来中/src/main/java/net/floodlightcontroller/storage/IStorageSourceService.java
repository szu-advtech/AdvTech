package net.floodlightcontroller.storage;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Future;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IStorageSourceService extends IFloodlightService {
    public void setTablePrimaryKeyName(String tableName, String primaryKeyName);
    void createTable(String tableName, Set<String> indexedColumns);
    Set<String> getAllTableNames();
    IQuery createQuery(String tableName, String[] columnNames, IPredicate predicate, RowOrdering ordering);
    IResultSet executeQuery(IQuery query);
    IResultSet executeQuery(String tableName, String[] columnNames, IPredicate predicate,
            RowOrdering ordering);
    Object[] executeQuery(String tableName, String[] columnNames, IPredicate predicate,
            RowOrdering ordering, IRowMapper rowMapper);
    void insertRow(String tableName, Map<String,Object> values);
    void updateRows(String tableName, List<Map<String,Object>> rows);
    void updateMatchingRows(String tableName, IPredicate predicate, Map<String,Object> values);
    void updateRow(String tableName, Object rowKey, Map<String,Object> values);
    void updateRow(String tableName, Map<String,Object> values);
    void deleteRow(String tableName, Object rowKey);
    void deleteRows(String tableName, Set<Object> rowKeys);
    void deleteMatchingRows(String tableName, IPredicate predicate);
    IResultSet getRow(String tableName, Object rowKey);
    void setExceptionHandler(IStorageExceptionHandler exceptionHandler);
    public Future<IResultSet> executeQueryAsync(final IQuery query);
    public Future<IResultSet> executeQueryAsync(final String tableName,
            final String[] columnNames,  final IPredicate predicate,
            final RowOrdering ordering);
    public Future<Object[]> executeQueryAsync(final String tableName,
            final String[] columnNames,  final IPredicate predicate,
            final RowOrdering ordering, final IRowMapper rowMapper);
    public Future<?> insertRowAsync(final String tableName, final Map<String,Object> values);
    public Future<?> updateRowsAsync(final String tableName, final List<Map<String,Object>> rows);
    public Future<?> updateMatchingRowsAsync(final String tableName, final IPredicate predicate,
            final Map<String,Object> values);
    public Future<?> updateRowAsync(final String tableName, final Object rowKey,
            final Map<String,Object> values);
    public Future<?> updateRowAsync(final String tableName, final Map<String,Object> values);
    public Future<?> deleteRowAsync(final String tableName, final Object rowKey);
    public Future<?> deleteRowsAsync(final String tableName, final Set<Object> rowKeys);
    public Future<?> deleteMatchingRowsAsync(final String tableName, final IPredicate predicate);
    public Future<?> getRowAsync(final String tableName, final Object rowKey);
    public Future<?> saveAsync(final IResultSet resultSet);
    public void addListener(String tableName, IStorageSourceListener listener);
    public void removeListener(String tableName, IStorageSourceListener listener);
    public void notifyListeners(List<StorageSourceNotification> notifications);
}
