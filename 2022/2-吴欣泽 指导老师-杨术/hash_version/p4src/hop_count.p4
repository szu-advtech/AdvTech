/*************************************************************************
    > File Name: hop_count.c
    > Author: 
    > Mail: 
    > Created Time: Fri 11 May 2018 9:12:19 AM CST
************************************************************************/

#include "includes/headers.p4"
#include "includes/parser.p4"

#define HOP_COUNT_SIZE 8
#define HC_BITMAP_SIZE 32
#define HC_COMPUTE_TABLE_SIZE 8
#define HC_COMPUTE_TWICE_TABLE_SIZE 3
#define TCP_SESSION_MAP_BITS 8
#define TCP_SESSION_MAP_SIZE 256 // 2^8
#define TCP_SESSION_STATE_SIZE 1
#define IP_TO_HC_INDEX_BITS 23
#define IP_TO_HC_TABLE_SIZE 8388608 // 2^23
#define SAMPLE_VALUE_BITS 3
#define PACKET_TAG_BITS 1

header_type meta_t {
    fields {
        hop_count: HOP_COUNT_SIZE; // Hop Count of this packet
        ip_to_hc_bitmap: HC_BITMAP_SIZE; // Hop Count bitmap in IP2HC
	ip_to_hc_bitmap2: HC_BITMAP_SIZE;
        tcp_session_map_index: TCP_SESSION_MAP_BITS;
        tcp_session_state: TCP_SESSION_STATE_SIZE; // 1:received SYN-ACK 0: exist or none
        tcp_session_seq: 32; // sequince number of SYN-ACK packet
        ip_to_hc_index : IP_TO_HC_INDEX_BITS;
	ip_to_hc_index2 : IP_TO_HC_INDEX_BITS;
        sample_value : SAMPLE_VALUE_BITS; // Used for sample packets
        hcf_state: 1; // 0: Learning 1: Filtering
        packet_tag: PACKET_TAG_BITS; // 0: Normal 1: Abnormal
        is_inspected: 1; // 0: Not Inspected 1: Inspected
	update_num : IP_TO_HC_INDEX_BITS; // update ip_hc2 number
    }
}

metadata meta_t meta;



//add 3 register to debug
register TCP_SESSION_line {
    width : TCP_SESSION_MAP_BITS;
    instance_count : 1;
}

register reg_sample_value {
    width : SAMPLE_VALUE_BITS;
    instance_count : 1;
}

register reg_ip_to_hc_index {
    width : IP_TO_HC_INDEX_BITS;
    instance_count : 1;
}


// The state of the switch, maintained by CPU(control.py)
register current_state {
    width : 1;
    instance_count : 1;
}

// The number(sampled) of abnormal packet per period
counter abnormal_counter {
    type : packets;
    instance_count : 1;
}

action check_hcf(is_inspected) {
    register_read(meta.hcf_state, current_state, 0);
    modify_field(meta.is_inspected, is_inspected);
}

// Used to get state(0:learning 1:filtering) of switch
// and judge whether the packet should be inspect by HCF
table hcf_check_table {
    reads {
        standard_metadata.ingress_port : exact;
    }
    actions { check_hcf; }
}

action sample_packet() {
    modify_field_rng_uniform(meta.sample_value, 0, (2 ^ SAMPLE_VALUE_BITS) - 1); //bug 1 
    register_write(reg_sample_value, 0, meta.sample_value); //!!!
}

// Sample packets for computing abnormal packets per period
table packet_sample_table {
    actions { sample_packet; }
}

action _drop() {
    drop();
}

action _nop() {
}

action tag_normal() {
    modify_field(meta.packet_tag, 0);
}

// Tag the packet as normal
table packet_normal_table {
    actions { tag_normal; }
}

action tag_abnormal() {
    modify_field(meta.packet_tag, 1);
}

// Tag the packet as abnormal
table packet_abnormal_table {
    actions { tag_abnormal; }
}

action compute_hc(initial_ttl) {
    subtract(meta.hop_count, initial_ttl, ipv4.ttl);
}

// According to final TTL, select initial TTL and compute hop count
table hc_compute_table {
    reads {
        ipv4.ttl : range;
    }
    actions {
        compute_hc;
    }
    size: HC_COMPUTE_TABLE_SIZE;
}

// Another for different pipeline
table hc_compute_table_copy {
    reads {
        ipv4.ttl : range;
    }
    actions {
        compute_hc;
    }
    size: HC_COMPUTE_TABLE_SIZE;
}

// The relation table between source IP and hop count
// Now, IP2HC use source IP's 23bit-prefix's hash value as index 
// and store hop-count's bit-map(Most hop count smaller than 30)
register ip_to_hc {
    width : HC_BITMAP_SIZE;
    instance_count : IP_TO_HC_TABLE_SIZE;
}

// 原文的时间维度表更新实现
register ip_to_hc2 { //缓存表
    width : HC_BITMAP_SIZE;
    instance_count : IP_TO_HC_TABLE_SIZE;
}
register reg_update_num {
    width : IP_TO_HC_INDEX_BITS; //23 bit
    instance_count : 1;
}


register reg_record_ip_to_hc {  //record to update
    width : IP_TO_HC_INDEX_BITS;
    instance_count : IP_TO_HC_TABLE_SIZE;
}


field_list ipsrc_hash_fields {
    ipv4.srcAddr;
}

field_list_calculation ipsrc_map_hash {
    input {
        ipsrc_hash_fields;
    }
    algorithm : crc16;
    output_width : IP_TO_HC_INDEX_BITS;
}

//两次哈希
field_list_calculation ipsrc_map_hash2 {
    input {
        ipsrc_hash_fields;
    }
    algorithm : crc32;
    output_width : IP_TO_HC_INDEX_BITS;
}
//


action inspect_hc() {
    modify_field_with_hash_based_offset(
        meta.ip_to_hc_index, 0,
        ipsrc_map_hash, IP_TO_HC_TABLE_SIZE
    );
    register_read(meta.ip_to_hc_bitmap, ip_to_hc, meta.ip_to_hc_index);
    //improvement
    register_write(ip_to_hc, meta.ip_to_hc_index, meta.ip_to_hc_bitmap);
    register_write(ip_to_hc2, meta.ip_to_hc_index, meta.ip_to_hc_bitmap);
    register_read(meta.update_num, reg_update_num, 0); // 设置更新数目
    register_write(reg_update_num, 0, meta.update_num+1);

    register_write(reg_record_ip_to_hc, meta.update_num+1, meta.ip_to_hc_index);// add to register

    //2nd Hash
    modify_field_with_hash_based_offset(
        meta.ip_to_hc_index2, 0,
        ipsrc_map_hash2, IP_TO_HC_TABLE_SIZE
    );
    register_read(meta.ip_to_hc_bitmap2, ip_to_hc, meta.ip_to_hc_index2);
    register_write(ip_to_hc, meta.ip_to_hc_index, meta.ip_to_hc_bitmap2);
    register_write(ip_to_hc2, meta.ip_to_hc_index, meta.ip_to_hc_bitmap2);
    register_read(meta.update_num, reg_update_num, 0);
    register_write(reg_update_num, 0, meta.update_num+1);
    register_write(reg_record_ip_to_hc, meta.update_num+1, meta.ip_to_hc_index);

}

// Get the origin hop count of this source IP
table hc_inspect_table {
    actions { inspect_hc; }
}

// Because near initial TTL pair: (30, 32) and (60, 64),
// so re-select initial TTL and re-compute hop count
table hc_compute_twice_table {
    reads {
        ipv4.ttl : range;
    }
    actions {
        _nop;
        compute_hc;
    }
    size: HC_COMPUTE_TABLE_SIZE;
}

action learning_abnormal() {
    count(abnormal_counter, 0);

    //如果想要学习非TCP数据包的跳数，可以去掉这段注释
    /*modify_field_with_hash_based_offset(
        meta.ip_to_hc_index, 0,
        ipsrc_map_hash, IP_TO_HC_TABLE_SIZE
    );
    register_read(meta.ip_to_hc_bitmap, ip_to_hc, meta.ip_to_hc_index);
    bit_or(meta.ip_to_hc_bitmap, meta.ip_to_hc_bitmap, 1 << meta.hop_count);
    register_write(ip_to_hc, meta.ip_to_hc_index, meta.ip_to_hc_bitmap);*/


    tag_normal();

}

action filtering_sample_abnormal() {
    count(abnormal_counter, 0);
    tag_abnormal();
}

action filtering_other_abnormal() {
    tag_abnormal();
}

// If the packet is judged as abnormal because its suspected hop-count,
// handle it according to the switch state and whether the packet is sampled.
// For learning state: if the packet is sampled, just update abnormal_counter 
// and tag it as normal(don't drop it); if the packet is not sampled, it won't 
// go through this table because switch don't check its hop count.
// For filtering state, every abnormal packets should be dropped but update 
// abnormal_counter specially for these sampled.
table hc_abnormal_table {
    reads {
        meta.hcf_state : exact;
        meta.sample_value : exact;
    }
    actions {
        learning_abnormal;
        filtering_sample_abnormal;
        filtering_other_abnormal;
    }
}

field_list l3_hash_fields {
    ipv4.srcAddr;
    ipv4.dstAddr;
    ipv4.protocol;
    tcp.srcPort;
    tcp.dstPort;
}

field_list_calculation tcp_session_map_hash {
    input {
        l3_hash_fields;
    }
    algorithm : crc16;
    output_width : TCP_SESSION_MAP_BITS;
}

field_list reverse_l3_hash_fields {
    ipv4.dstAddr;
    ipv4.srcAddr;
    ipv4.protocol;
    tcp.dstPort;
    tcp.srcPort;
}

field_list_calculation reverse_tcp_session_map_hash {
    input {
        reverse_l3_hash_fields;
    }
    algorithm : crc16;
    output_width : TCP_SESSION_MAP_BITS;
}

action lookup_session_map() {
    modify_field_with_hash_based_offset(
        meta.tcp_session_map_index, 0,
        tcp_session_map_hash, TCP_SESSION_MAP_SIZE
    );
    //register_write(TCP_SESSION_line, 0, meta.tcp_session_map_index); // my test
    register_read(
        meta.tcp_session_state, session_state, 
        meta.tcp_session_map_index
    );
    register_read(
        meta.tcp_session_seq, session_seq,
        meta.tcp_session_map_index
    );
    register_write(TCP_SESSION_line, 0, tcp.ackNo); //my test 看一下ackNo是否传过来了
							//我设置的这个TCP_SESSION_line寄存器就是可以用来测试的，只要比特位够就行
}

action lookup_reverse_session_map() {
    modify_field_with_hash_based_offset(
        meta.tcp_session_map_index, 0,
        reverse_tcp_session_map_hash,
        TCP_SESSION_MAP_SIZE
    );
    register_read(
        meta.tcp_session_state, session_state, 
        meta.tcp_session_map_index
    );
    register_read(
        meta.tcp_session_seq, session_seq,
        meta.tcp_session_map_index
    );
}

// Get packets' tcp session information. Notice: dual direction packets in one 
// flow should belong to same tcp session and use same hash value
table session_check_table {
    reads {
        standard_metadata.ingress_port : exact;
    }
    actions {
        _drop;
        lookup_session_map;
        lookup_reverse_session_map;
    }
}

// Store sesscon state for concurrent tcp connections
register session_state {
    width : TCP_SESSION_STATE_SIZE;
    instance_count : TCP_SESSION_MAP_SIZE;
} 

// Store sesscon sequince number(SYN-ACK's) for concurrent tcp connections
register session_seq {
    width : 32;
    instance_count : TCP_SESSION_MAP_SIZE;
} 

action init_session() {
    register_write(session_state, meta.tcp_session_map_index, 1);
    register_write(session_seq, meta.tcp_session_map_index, tcp.seqNo);
}

// Someone is attempting to establish a connection from server
table session_init_table {
    actions {
        init_session;
    }
}

action complete_session() {
    register_write(session_state, meta.tcp_session_map_index, 0);
    modify_field_with_hash_based_offset(
        meta.ip_to_hc_index, 0,
        ipsrc_map_hash, IP_TO_HC_TABLE_SIZE
    );
    register_read(meta.ip_to_hc_bitmap, ip_to_hc, meta.ip_to_hc_index);
    bit_or(meta.ip_to_hc_bitmap, meta.ip_to_hc_bitmap, 1 << meta.hop_count);
    register_write(ip_to_hc, meta.ip_to_hc_index, meta.ip_to_hc_bitmap);
    //improvement
    register_write(ip_to_hc2, meta.ip_to_hc_index, meta.ip_to_hc_bitmap);
    register_write(reg_ip_to_hc_index, 0, meta.ip_to_hc_index); //my test
    register_read(meta.update_num, reg_update_num, 0); // 设置更新数目
    register_write(reg_update_num, 0, meta.update_num+1);
    register_write(reg_record_ip_to_hc, meta.update_num+1, meta.ip_to_hc_index);// add to register

    //2nd Hash
    modify_field_with_hash_based_offset(
        meta.ip_to_hc_index2, 0,
        ipsrc_map_hash2, IP_TO_HC_TABLE_SIZE
    );
    register_read(meta.ip_to_hc_bitmap2, ip_to_hc, meta.ip_to_hc_index2);
    bit_or(meta.ip_to_hc_bitmap2, meta.ip_to_hc_bitmap2, 1 << meta.hop_count);
    register_write(ip_to_hc, meta.ip_to_hc_index2, meta.ip_to_hc_bitmap2);
    register_write(ip_to_hc2, meta.ip_to_hc_index2, meta.ip_to_hc_bitmap2);
    register_read(meta.update_num, reg_update_num, 0); 
    register_write(reg_update_num, 0, meta.update_num+1);
    register_write(reg_record_ip_to_hc, meta.update_num+1, meta.ip_to_hc_index);

    tag_normal();
}

// Establish the connection, and update IP2HC
table session_complete_table {
    reads {
        tcp.ack : exact;
    }
    actions {
        tag_abnormal;
        complete_session;
    }
}

action forward_l2(egress_port) {
    modify_field(standard_metadata.egress_spec, egress_port);
}

// Forward table, now it just support layer 2
table l2_forward_table {
    reads {
        meta.packet_tag : exact;
        standard_metadata.ingress_port : exact;
    }
    actions {
        _drop;
        forward_l2;
    }
}


control ingress {
    // Get basic infomation of switch and tcp session
    apply(hcf_check_table);   //判断current_state，设置检查位，默认1检查，端口2发出的不查
    apply(session_check_table);  //根据五元组哈希查session状态及seq
    if (meta.tcp_session_state == 1) {
        // The connection is wainting to be established
        if (tcp.ackNo == meta.tcp_session_seq + 1) {
            // Legal connection, so real hop count to be stored
            apply(hc_compute_table_copy); //计算跳数
            apply(session_complete_table); //session状态置0，ip2hc表中（行号为为源ip哈希值，列号为跳数）的位更新
        }
        else {
            // Illegal connection attempt
            apply(packet_abnormal_table);
        }
    }
    else if (meta.tcp_session_state == 0) {
        // TCP session has been established or not
        if (tcp.syn == 1 and tcp.ack == 1) {
            // A client is attempting to connect to the server
            apply(session_init_table);
        }
        else if (meta.is_inspected == 1) {
            // Other packets. Anyway, samplet it first
            apply(packet_sample_table);
            if (meta.sample_value == 0 or meta.hcf_state == 1) {
                // The packet is sampled or the switch is in filtering state
                // Compute packet's hop count and refer to its origin hop count
                apply(hc_compute_table);
                apply(hc_inspect_table);
                if (((meta.ip_to_hc_bitmap >> meta.hop_count) & 1) == 1 and ((meta.ip_to_hc_bitmap2 >> meta.hop_count) & 1) == 1) {
                    apply(packet_normal_table);
		}
		else {
                    apply(hc_compute_twice_table);
                    if (((meta.ip_to_hc_bitmap >> meta.hop_count) & 1) == 1 and ((meta.ip_to_hc_bitmap2 >> meta.hop_count) & 1) == 1) {
                        apply(packet_normal_table);
                        
                    }
                    else {
                        apply(hc_abnormal_table);
                    }
                }

            }
            else {
                // Do nothing to these packets
                apply(packet_normal_table);
            }
        }
    }
    // Drop abnormal packets and forward normal packets in layer two
    apply(l2_forward_table);
}

control egress {
}
