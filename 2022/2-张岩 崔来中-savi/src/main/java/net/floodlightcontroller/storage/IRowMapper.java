package net.floodlightcontroller.storage;
public interface IRowMapper {
    Object mapRow(IResultSet resultSet);
}
