pragma solidity ^0.8.0;


contract SLA {
    uint public fee_service;
    uint public fee_compensation;
    uint public fee_witness;
    uint public N;//the number of witness
    uint public M;
    address payable public customer;
    address payable public provider;
    address payable public witness;

    //枚举类型
    enum State { Fresh, Init, Active, Completed, Violated }

    State public state;

    //支付条件
    modifier condition(bool condition_) {
        require(condition_);
        _;
    }

    /// Only the customer can call this function.
    error OnlyCustomer();
    /// Only the provider can call this function.
    error OnlyProvider();
    /// Only the witness can call this function.
    error OnlyWitness();
    /// The function cannot be called at the current state.
    error InvalidState();
    /// The provided value has to be even.
    error ValueNotEven();

    //modifier用于在函数执行前检查某种前置条件，是一种合约属性
    modifier onlyCostumer() {
        require(
            msg.sender == customer,//msg.sender代表合约调用者地址
            "Only customer can call this."
        );//require用于检查输入变量是否满足条件,".."中是返回值，返回信息需要消耗gas
        _;
    }
    modifier onlyProvider() {
        require(
            msg.sender == provider,
            "Only provider can call this."
        );
        _;
    }

    modifier onlyWitness() {
        require(
            msg.sender == witness,
            "Only witness can call this."
        );
        _;
    }

    modifier inState(State _state) {
        require(
            state == _state,
            "Invalid state."
        );
        _;
    }

    //EVM日志基础设施提供的接口，事件发生时会触发参数存储到交易日志中。
    //这些日志与合约地址关联，并记录到区块链中
    //使用emit来触发事件
    event SetCustomer();
    event SortitionRequested();
    event WitnessConfirmed();
    event SLASetup();
    event SLAAccepted();
    event SLACanceled();
    event ViolationReported();
    event ViolationConfirmed();
    event WitnessReset();
    event VSLAEnded();
    event NSLAEnded();
    event WitnessWithdraw();
    event ProviderWithdraw();
    event CustomerWithdraw();
    event SLAReset();
    event SLARestarted();
    event eventFallback(string);

    /*constructor构造函数，仅在部署时调用一次，用于进行状态变量的初始化工作
    constructor() payable {
        provider = payable(msg.sender);
        state = State.Fresh;
    }*/

    //provider选择off-chain协商好的customer,setCustomer,publishService
    function SLAcreator(address _provider, address _customer, address _witness, uint _feeService,
        uint _feeCompensation, uint _feeWitness, uint _N, uint _M)
    payable
    external
    returns (address _addr)
    {
        emit SetCustomer();
        provider = payable(_provider);
        //_addr = address(this);
        //provider向SLA合约账户转入证人费用
        //payable(_addr).transfer(_feeWitness);
        state = State.Fresh;
        customer = payable(_customer);
        witness = payable(_witness);
        //number of witnessN, M, fee of service, compensation, fee of witness, time of service
        fee_service = _feeService;
        fee_compensation = _feeCompensation;
        fee_witness = _feeWitness;
        N = _N;
        M = _M;
    }

    //Refresh

    //witness committee sortition request
    function sortitionRequest()
    internal
    {
        emit SortitionRequested();
    }

    function setupSLA()
    external
    onlyProvider
    inState(State.Fresh)
    {
        emit SLASetup();
        state = State.Init;
    }
    //requestSortition
    //WitnessConfirm

    //Init
    //cancelSLA （Time Window
    //acceptSLA
    function acceptSLA()
    external
    onlyCostumer
    condition(msg.value == (fee_service + fee_witness))
    inState(State.Init)
    payable
    {
        emit SLAAccepted();
        state = State.Active;
    }

    //completed
    function providerWithdraw(uint value_p)
    inState(State.Completed)
    public
    payable
    {
        emit ProviderWithdraw();
        provider.transfer(value_p);//value_p
    }

    function witnessWithdraw()
    inState(State.Completed)
    public
    payable
    {
        emit WitnessWithdraw();
        witness.transfer(2 * fee_witness);//value_w
    }

    //Active
    //Service Duration End_out
    function providerEndNSLA()
    external
    onlyProvider
    inState(State.Active)
    {
        emit NSLAEnded();
        state = State.Completed;

        providerWithdraw(fee_service);
        witnessWithdraw();
    }

    //Service Duration End_in
    function reportViolation()
    external
    onlyWitness
    inState(State.Active)
    {
        emit ViolationReported();
        //confirmViolation
    }

    function resetWitness()
    external
    onlyCostumer
    inState(State.Active)
    {
        //resetWitness
        //service
    }

    //ViolationConfirmed
    function confirmViolation()
    external
    onlyCostumer
    inState(State.Active)
    {
        emit ViolationConfirmed();
        state   = State.Violated;
    }

    //violated
    function customerEndVSLA()
    external
    payable
    onlyCostumer
    inState(State.Violated)
    {
        emit VSLAEnded();
        state = State.Completed;
        customer.transfer(fee_compensation);
        providerWithdraw(fee_service - fee_compensation);
        witnessWithdraw();
    }




    fallback() external payable{
        emit eventFallback('fallback');
    }

    receive() external payable{
        emit eventFallback('receive');
    }
}

contract WitnessPool {
    uint public oc;//在线的用户人数
    uint public K_s;
    uint public K_c;

    address public customer;
    address public provider;

    address public SC;
    //枚举类型
    enum State { Offline, Online, Candidate, Busy }

    //证人结构
    struct Witness {
        bool report;  // 若为真，代表该人报告违规
        address witness_address; // 证人地址
        uint index;   // 在证人池列表中的索引
    }
    //证人池中的用户结构
    struct User {
        State state;
        uint reputation; //证人信誉
    }

    /// The function cannot be called at the current state.
    error WitnessInvalidState();

    modifier inState(State _state) {
        require(
            users[msg.sender].state == _state,
            "witeness Invalid state."
        );
        _;
    }

    event OneUserRegistered();
    event UserTurnedOn();
    event UserTurnedOff();
    event SLAGenerated();
    event Sorted();
    event OneWitnessSorted();
    event WitnessReversed();
    event WitnessRejected();
    event WitnessConfirmed();
    event WitnessCommitteeRelease();
    event SLAStateCheck(State);
    event AddressCheck(address);
    event BalanceCheck(uint);

    // 一个 address类型的动态数组
    address[] public WP;
    address[] public SW;

    // 声明一个状态变量，为每个用户地址存储一个 `User`。
    mapping(address => User) public users;

    constructor(){
        K_s = 10;
        K_c = 12;
    }

    //对于每个注册的用户创建一个新的 User 对象并把它添加到数组的末尾。
    function register() external {
        emit OneUserRegistered();
        address register_one = msg.sender;
        WP.push(register_one);

        User storage sender = users[register_one];
        sender.state = State.Offline;
        sender.reputation = 5;
    }
    //offline
    function userTurnOn()
    inState(State.Offline)
    external
    {
        emit UserTurnedOn();
        users[msg.sender].state = State.Online;
        oc++;//在线人数+1
    }
    //online
    function userTurnOff()
    inState(State.Online)
    external
    {
        emit UserTurnedOff();
        users[msg.sender].state = State.Offline;
        oc--;
    }

    function getblance(address payable addr)public payable{
        uint test1 = addr.balance;
        emit BalanceCheck(test1);
    }

    //Two users negotiate off-chain.
    //The provider use this to be a provider and generate a SLA
    function generateSLA(address _customer, address _witness, uint _feeService,
        uint _feeCompensation, uint _feeWitness, uint _N, uint _M)
    external
    payable
    inState(State.Online)
    {
        SLA _sla = new SLA();
        address _slaAddress = address(_sla);
        emit AddressCheck(_slaAddress);

        provider = payable(msg.sender);
        payable(_slaAddress).transfer(_feeWitness);
        //_sla.state = SLA.State.Fresh;
        users[_witness].state = State.Candidate;

        _sla.SLAcreator(msg.sender, _customer, _witness, _feeService, _feeCompensation, _feeWitness, _N, _M);

    }


    //SLA的sortition接口被调用后的K=22个块后再调用该接口
    function sortition(uint B_b, uint N, address customer, address provider)
    external
        //returns (address[] memory SW)
    {
        require(block.number > (B_b + K_s + K_c));
        require(oc >= (10 * N));

        emit Sorted();

        uint seed = 0;
        for(uint i = 0; i < K_s ; i++){
            bytes32 _hash = blockhash(B_b + i + 1);
            uint _uint_hash = uint(_hash);
            seed += _uint_hash;
        }

        uint j = 0;
        uint index;
        uint len = WP.length;
        while(j < N){
            index = seed % len;
            address _address = WP[index];
            User storage _index = users[_address];
            emit SLAStateCheck(_index.state);
            if ( _index.state == State.Online
            && _index.reputation > 0
            && _address != customer
                && _address != provider
            ){
                emit OneWitnessSorted();
                _index.state = State.Candidate;
                oc--;
                SW[j] = _address;
                j++;
                //发送证人确认窗口，开始窗口计时
            }
            bytes32 _bytes32Hashseed = keccak256(abi.encodePacked(seed));
            seed = uint(_bytes32Hashseed);
        }//candidates sorted
    }

    //确认窗口超时
    function witnessReverse(address _timeout)
    external
    {
        User storage _timeoutUser = users[_timeout];
        require(_timeoutUser.state == State.Candidate);
        emit WitnessReversed();

        _timeoutUser.state = State.Online;
        _timeoutUser.reputation --;
    }

    function witnessReject()
    external
    inState(State.Candidate)
    {
        emit WitnessRejected();
        users[msg.sender].state = State.Online;
    }

    //comfirm->SLA
    function witnessConfirm()
    external
    inState(State.Candidate)
    {
        emit WitnessConfirmed();
        users[msg.sender].state = State.Busy;
    }

    //SLA -> WPRelease
    function witnessRelease()
    external
    {
        emit WitnessCommitteeRelease();
        users[msg.sender].state = State.Online;
    }

}