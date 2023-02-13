/* eslint-disable no-continue */
/* eslint-disable no-lonely-if */
/* eslint-disable no-prototype-builtins */
/* eslint-disable no-restricted-syntax */
/* eslint-disable radix */
/* eslint-disable no-restricted-properties */
/* eslint-disable operator-linebreak */
/* eslint-disable prefer-destructuring */
/* eslint-disable prefer-template */
/* eslint-disable no-param-reassign */
/* eslint-disable brace-style */
/* eslint-disable dot-notation */
/* eslint-disable camelcase */
/* eslint-disable no-plusplus */
/* eslint-disable no-underscore-dangle */
/* eslint-disable consistent-return */
/*
Copyright 2016 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
const common = require('./common');

class Analyze {
  constructor() {
    this.traceEvents = [];
    this.loadingTraceEvents = [];
    this.renderingTraceEvents = [];
    this.paintingTraceEvents = [];
    this.networkTraceEvents = [];
    this.netlogTraceEvents = [];
    this.frameEvents = [];
    this.cpu = { main_thread: null };
    this.network = {};
    this.loading = {};
    this.rendering = {};
    this.painting = {};
    this.threads = {};
    this.ignoreThreads = {};
    this.startTime = null;
    this.endTime = null;
    this.threadStack = {};
    this.eventNames = {};
    this.eventNameLookup = {};
    this.timelineEvents = [];
    this.script = null;
    this.scriptXtra = {};
    this.userTiming = [];
    this.networkList = [];
    this.loadingList = [];
    this.scriptList = [];
    this.ordered = {};
    this.orderedUrlLookup = {};
    this.networkLookupUrl = {};
    this.networkLookupId = {};
    this.loadingLookupUrl = {};
    this.loadingLookupId = {};
    this.scriptLookupUrl = {};
    this.scriptLookupId = {};
    this.renderingList = [];
    this.paintingList = [];
    this.netlog = { bytes_in: 0, bytes_out: 0, next_request_id: 1000000 };
    this.netlogRequests = null;
    this.all = [];
    this.parse0Id = null;
    this.download0Id = null;
    this.output = [];
    this.output2 = {};
    this.allStartTimeLookup = {};
    this.lastActivity = [];
    this.allDict = {};
    this.deps = [];
    this.criticalPath = [];
    this.timeToInteractive = -1000;
    this.interactiveStart = 0;
    this.interactiveEnd = null;
    this.interactive = [];
    this.depsParent = {};
    this.depsNext = {};
    this.firstMeaningfulPaintDict = {};
    this.firstMeaningfulPaintCandidatesDict = {};
    this.firstContentfulPaintDict = {};
    this.firstPaintDict = {};
    this.loadEventEndDict = {};
    this.domContentLoadedEventEndDict = {};
    this.firstMeaningfulPaintList = [];
    this.firstMeaningfulPaintCandidatesList = [];
    this.firstContentfulPaintList = [];
    this.firstPaintList = [];
    this.loadEventEndList = [];
    this.domContentLoadedEventEndList = [];
    this.firstMeaningfulPaint = -1000;
    this.firstContentfulPaint = -1000;
    this.firstPaint = -1000;
    this.loadEventEnd = -1000;
    this.domContentLoadedEventEnd = -1000;
    this.networkingCp = 0.0;
    this.scriptingCp = 0.0;
    this.loadingCp = 0.0;
    this.networkingTotal = 0;
    this.scriptingTotal = 0;
    this.loadingTotal = 0;
    this.mainFrameId = '';
    this.fromScriptSet = new Set([]);
    this.javascriptTypes = ['application/x-javascript', 'application/javascript', 'application/ecmascript',
      'text/javascript', 'text/ecmascript', 'application/json', 'javascript/text'];
    this.cssTypes = ['text/css', 'css/text'];
    this.textTypes = ['evalhtml', 'text/html', 'text/plain', 'text/xml'];
  }

  async analyzeTrace(traceEvents) {
    try {
      await this.processTrace(traceEvents);
      await this.sortEventsByTs();
      await this.processNetworkEvents(this.networkTraceEvents);
      await this.processLoadingEvents(this.loadingTraceEvents);
      await this.processRenderingEvents(this.renderingTraceEvents);
      await this.processPaintingEvents(this.paintingTraceEvents);
      await this.findInterestingTimes();
      await this.sortAllProcessedByStartTime();
      await this.extractDependencies();
      await this.findCriticalPath(this.lastActivity[0][0]);
      await this.criticalPath.reverse();

      if (this.criticalPath[0] === null) {
        this.criticalPath.splice(0, 1);
      }
      await this.timeSummary();
      await this.orderLayout();

      return this.writeOutputlog();
    } catch (error) {
      console.error(error);
    }
  }

  processTrace(traceEvents) {
    const _this = this;
    return new Promise((resolve, reject) => {
      for (let i = 0; i < traceEvents.length; i++) {
        const _traceEvent = traceEvents[i];
        try {
          _this.filterTraceEvents(_traceEvent);
        } catch (err) {
          // reject(err);
          console.log(err);
        }
      }
      _this.processTraceEvents();
      resolve();
    });
  }

  filterTraceEvents(trace_event) {
    const { cat, name } = trace_event;
    if (cat === 'toplevel' || cat === 'ipc,toplevel') {
      return;
    }

    if ('args' in trace_event && 'data' in trace_event['args']) {
      if ('url' in trace_event['args']['data']) {
        if (trace_event['args']['data']['url'].includes('bperf.xyz')) {
          return;
        }
      }
    }

    if (name === 'TracingStartedInPage'
      && ('args' in trace_event)
      && ('data' in trace_event['args'])
      && ('page' in trace_event['args']['data'])) {
      this.mainFrameId = trace_event['args']['data']['page'];
    }
    else if (name === 'TracingStartedInBrowser'
      && ('args' in trace_event)
      && ('data' in trace_event['args'])
      && ('frames' in trace_event['args']['data'])) {
      this.mainFrameId = trace_event['args']['data']['frames'][0]['frame'];
    }
    /*
    If there was no firstMeaningfulPaint event found in the trace In this case,
    we'll use the last firstMeaningfulPaintCandidate we can find.
    firstContentfulPaint
    firstPaint
    loadEventEnd (for this frameId ) use last
    domContentLoadedEventEnd (for this frameId) use last
    */
    if (name === 'firstMeaningfulPaint' && 'args' in trace_event && 'frame' in trace_event['args']) {
      const _frameId = trace_event['args']['frame'];
      if (!(_frameId in this.firstMeaningfulPaintDict)) {
        this.firstMeaningfulPaintDict[_frameId] = [trace_event['ts']];
      }
      else {
        this.firstMeaningfulPaintDict[_frameId].push(trace_event['ts']);
      }
    }

    if (name === 'firstMeaningfulPaintCandidate' && 'args' in trace_event && 'frame' in trace_event['args']) {
      const _frameId = trace_event['args']['frame'];
      if (!(_frameId in this.firstMeaningfulPaintCandidatesDict)) {
        this.firstMeaningfulPaintCandidatesDict[_frameId] = [trace_event['ts']];
      }
      else {
        this.firstMeaningfulPaintCandidatesDict[_frameId].push(trace_event['ts']);
      }
    }

    else if (name === 'firstContentfulPaint' && 'args' in trace_event && 'frame' in trace_event['args']) {
      const _frameId = trace_event['args']['frame'];
      if (!(_frameId in this.firstContentfulPaintDict)) {
        this.firstContentfulPaintDict[_frameId] = [trace_event['ts']];
      }
      else {
        this.firstContentfulPaintDict[_frameId].push(trace_event['ts']);
      }
    }

    if (name === 'firstPaint' && 'args' in trace_event && 'frame' in trace_event['args']) {
      const _frameId = trace_event['args']['frame'];
      if (!(_frameId in this.firstPaintDict)) {
        this.firstPaintDict[_frameId] = [trace_event['ts']];
      }
      else {
        this.firstPaintDict[_frameId].push(trace_event['ts']);
      }
    }

    if (name === 'loadEventEnd' && 'args' in trace_event && 'frame' in trace_event['args']) {
      const _frameId = trace_event['args']['frame'];
      if (!(_frameId in this.loadEventEndDict)) {
        this.loadEventEndDict[_frameId] = [trace_event['ts']];
      }
      else {
        this.loadEventEndDict[_frameId].push(trace_event['ts']);
      }
    }

    if (name === 'domContentLoadedEventEnd' && 'args' in trace_event && 'frame' in trace_event['args']) {
      const _frameId = trace_event['args']['frame'];
      if (!(_frameId in this.domContentLoadedEventEndDict)) {
        this.domContentLoadedEventEndDict[_frameId] = [trace_event['ts']];
      }
      else {
        this.domContentLoadedEventEndDict[_frameId].push(trace_event['ts']);
      }
    }
    if (cat === 'devtools.timeline'
      || cat.indexOf('devtools.timeline') !== -1
      || cat.indexOf('blink.feature_usage') !== -1
      || cat.indexOf('blink.user_timing') !== -1) {
      this.processXtraScriptEvents(trace_event);
      this.traceEvents.push(trace_event);
    }
    if ((cat === 'devtools.timeline' && name === 'ParseHTML') || (
      cat === 'blink,devtools.timeline' && name === 'ParseAuthorStyleSheet')) {
      this.loadingTraceEvents.push(trace_event);
    }
    if ((cat === 'disabled-by-default-devtools.timeline'
      || cat.indexOf('devtools.timeline') !== -1)
      && (['CompositeLayers', 'Paint'].includes(name))) {
      this.paintingTraceEvents.push(trace_event);
    }
    if ((cat === 'devtools.timeline' || cat.indexOf('devtools.timeline') !== -1)
      && (['Layout', 'UpdateLayerTree', 'HitTest', 'RecalculateStyles'].includes(name))) {
      this.renderingTraceEvents.push(trace_event);
    }
    if (cat === 'devtools.timeline' && (['ResourceSendRequest',
      'ResourceReceiveResponse', 'ResourceReceivedData', 'ResourceFinish'].includes(name))) {
      this.networkTraceEvents.push(trace_event);
    }

    if (['TracingStartedInBrowser', 'TracingStartedInPage'].includes(name)) {
      this.frameEvents.push(trace_event);
    }
  }

  processXtraScriptEvents(trace_event) {
    const { name } = trace_event;
    if (name === 'EvaluateScript' && ('ph' in trace_event && trace_event['ph'] === 'X')) {
      if ('args' in trace_event && 'data' in trace_event['args']
        && 'url' in trace_event['args']['data']) {
        const _id = trace_event['args']['data']['url'].split('#')[0];
        if (_id.length > 0) {
          const _threadX = trace_event['pid'] + ':' + trace_event['tid'];
          if (!(_threadX in this.scriptXtra)) {
            this.scriptXtra[_threadX] = {};
          }
          const _dur = trace_event['dur'];
          const _ts = trace_event['ts'];
          if (!(_id in this.scriptXtra[_threadX])) {
            this.scriptXtra[_threadX][_id] = {};
            this.scriptXtra[_threadX][_id]['EvaluateScript'] = [[_dur, _ts]];
          }
          else {
            this.scriptXtra[_threadX][_id]['EvaluateScript'].push([_dur, _ts]);
          }
        }
      }
    }
  }

  processTraceEvents() {
    // sort the raw trace events by timestamp && then process them
    if (this.traceEvents.length >= 1) {
      this.traceEvents.sort((a, b) => a['ts'] - b['ts']);

      for (let i = 0; i < this.traceEvents.length; i++) {
        this.processTraceEvent(this.traceEvents[i]);
      }
      this.traceEvents = [];
    }
    // Post-process the netlog events (may shift the start time)
    this.processTimelineEvents();
  }

  processTraceEvent(trace_event) {
    const { cat } = trace_event;
    if (cat === 'devtools.timeline' || cat.indexOf('devtools.timeline') !== -1) {
      this.processTimelineTraceEvent(trace_event);
    }
    else if (cat.indexOf('blink.user_timing') !== -1) {
      this.userTiming.push(trace_event);
    }
  }


  processTimelineTraceEvent(trace_event) {
    const thread = trace_event['pid'] + ':' + trace_event['tid'];
    // Keep track of the main thread
    if (this.cpu['main_thread'] === null && trace_event['name'] === 'ResourceSendRequest') {
      if ('args' in trace_event && 'data' in trace_event['args']) {
        if ('url' in trace_event['args']['data']
          && trace_event['args']['data']['url'].startsWith('http')) {
          if (!(thread in this.threads)) {
            this.threads[thread] = {};
          }
          if ((this.startTime === null) || trace_event['ts'] < this.startTime) {
            this.startTime = trace_event['ts'];
          }
          this.cpu['main_thread'] = thread;
          if (!('dur' in trace_event)) {
            trace_event['dur'] = 1;
          }
        }
      }
    }
    // Make sure each thread has a numerical ID
    if ((this.cpu['main_thread'] !== null) && !(thread in this.threads) &&
      !(thread in this.ignoreThreads) && trace_event['name'] !== 'Program') {
      this.threads[thread] = {};
    }
    // Build timeline events on a stack. 'B' begins an event, 'E' ends an event
    if ((thread in this.threads) && ('dur' in trace_event
      || trace_event['ph'] === 'B' || trace_event['ph'] === 'E')) {
      trace_event['thread'] = this.threads[thread];
      if (!(thread in this.threadStack)) {
        this.threadStack[thread] = [];
      }
      if (!(trace_event['name'] in this.eventNames)) {
        this.eventNames[trace_event['name']] = Object.keys(this.eventNames).length;
        this.eventNameLookup[this.eventNames[trace_event['name']]] = trace_event['name'];
      }
      if (!(trace_event['name'] in this.threads[thread])) {
        this.threads[thread][trace_event['name']] = this.eventNames[trace_event['name']];
      }
      let e = null;
      if (trace_event['ph'] === 'E') {
        if (this.threadStack[thread].length > 0) {
          e = this.threadStack[thread].pop();
          if (e['n'] === this.eventNames[trace_event['name']]) {
            e['e'] = trace_event['ts'];
          }
        }
      }
      else {
        e = {
          t: thread,
          n: this.eventNames[trace_event['name']],
          s: trace_event['ts'],
        };
        if (((trace_event['name'] === 'EvaluateScript')
          || (trace_event['name'] === 'v8.compile')
          || (trace_event['name'] === 'v8.parseOnBackground'))
          && ('args' in trace_event
          && 'data' in trace_event['args']
          && 'url' in trace_event['args']['data']
          && trace_event['args']['data']['url'].startsWith('http'))) {
          e['js'] = trace_event['args']['data']['url'].split('#')[0];
        }
        if (trace_event['name'] === 'FunctionCall' && 'args' in trace_event &&
          'data' in trace_event['args'] &&
          'scriptName' in trace_event['args']['data'] &&
          trace_event['args']['data'][
            'scriptName'].startsWith('http')) {
          e['js'] = trace_event['args']['data']['scriptName'];
        }
        if (trace_event['ph'] === 'B') {
          this.threadStack[thread].push(e);
          e = null;
        }
        else if ('dur' in trace_event) {
          e['e'] = e['s'] + trace_event['dur'];
        }
      }
      if (e !== null && 'e' in e && e['s'] >= this.startTime && e['e'] >= e['s']) {
        if (this.endTime === null || e['e'] > this.endTime) {
          this.endTime = e['e'];
        }
        // attach it to a parent event if there is one
        if (this.threadStack[thread].length > 0) {
          const parent = this.threadStack[thread].pop();
          if (!('c' in parent)) {
            parent['c'] = [];
          }
          parent['c'].push(e);
          this.threadStack[thread].push(parent);
        }
        else {
          this.timelineEvents.push(e);
        }
      }
    }
  }

  processTimelineEvents() {
    if (this.timelineEvents.length >= 1 && this.endTime > this.startTime) {
      // Figure out how big each slice should be in usecs.
      // Size it to a power of 10 where we have at least 2000 slices
      let exp = 0;
      let last_exp = 0;
      let slice_count = this.endTime - this.startTime;
      while (slice_count > 2000) {
        last_exp = exp;
        exp += 1;
        slice_count = parseInt(Math.ceil(parseFloat(this.endTime -
          this.startTime) / parseFloat(Math.pow(10, exp))));
      }
      this.cpu['total_usecs'] = this.endTime - this.startTime;
      this.cpu['slice_usecs'] = parseInt(Math.pow(10, last_exp));
      slice_count = parseInt(Math.ceil(parseFloat(this.endTime -
        this.startTime) / parseFloat(this.cpu['slice_usecs'])));
      // Create the empty time slices for all of the threads
      this.cpu['slices'] = {};
      for (const thread in this.threads) {
        if (this.threads.hasOwnProperty(thread)) {
          this.cpu['slices'][thread] = { total: Array(slice_count).fill(0.0) };
          for (const name in this.threads[thread]) {
            if (this.threads[thread].hasOwnProperty(name)) {
              this.cpu['slices'][thread][name] = Array(slice_count).fill(0.0);
            }
          }
        }
      }
      // Go through all of the timeline events recursively and
      // account for the time they consumed
      var arrayLength2 = this.timelineEvents.length;
      for (let i = 0; i < arrayLength2; i++) {
        this.processTimelineEvent({ timeline_event: this.timelineEvents[i], parent: null, stack: null });
      }
      if (!(this.interactiveEnd === null) && this.interactiveEnd -
        this.interactiveStart > 500000) {
        this.interactive.push([parseInt(Math.ceil(this.interactiveStart / 1000.0)),
        parseInt(Math.floor(this.interactiveEnd / 1000.0))]);
      }

      // Go through all of the fractional times and convert the
      // parseFloat fractional times to parseInteger usecs
      for (var thread in this.cpu['slices']) {
        if (this.cpu.hasOwnProperty(thread)) {
          delete this.cpu['slices'][thread]['total'];
          for (var name in this.cpu['slices'][thread]) {
            if (this.cpu['slices'][thread].hasOwnProperty(name)) {
              //var sliceLength = this.cpu['slices'][thread][name].length;
              var sliceLength = Object.keys(this.cpu['slices'][thread][name]).length;
              for (var slice = 0; slice < sliceLength; slice++) {
                this.cpu['slices'][thread][name][slice] =
                  parseInt(this.cpu['slices'][thread][name][slice]
                    * this.cpu['slice_usecs']);
              }
            }
          }
        }
      }
    }
  }

  //
  processTimelineEvent({ timeline_event, parent, stack } = {}) {
    var start = timeline_event['s'] - this.startTime;
    var end = timeline_event['e'] - this.startTime;
    if (stack === null) {
      var stack = {};
    }
    if (end > start) {
      var elapsed = end - start;
      var thread = timeline_event['t'];
      var name = this.eventNameLookup[timeline_event['n']];
      //TODO If DOMContentLoadedEnd occurs after the start of the window, report that as TTI.
      // Keep track of periods on the main thread where at least 500ms are
      //available with no tasks longer than 50ms
      if ('main_thread' in this.cpu && thread === this.cpu['main_thread']) {
        if (elapsed > 50000) {
          if ((start - this.interactiveStart) > 500000) {
            this.interactive.push(
              [parseInt(Math.ceil(this.interactiveStart / 1000.0)),
              parseInt(Math.floor(start / 1000.0))]);
          }
          this.interactiveStart = end;
          this.interactiveEnd = null;
        }
        else {
          this.interactiveEnd = end;
        }
      }
      if ('js' in timeline_event) {
        var script = timeline_event['js'];
        var js_start = start / 1000.0;
        var js_end = end / 1000.0;
        if (this.script === null) {
          this.script = {};
        }
        if (!('main_thread' in this.script) && ('main_thread' in this.cpu)) {
          this.script['main_thread'] = this.cpu['main_thread'];
        }
        if (!(thread in this.script)) {
          this.script[thread] = {};
        }
        if (!(script in this.script[thread])) {
          this.script[thread][script] = {};
        }
        if (!(name in this.script[thread][script])) {
          this.script[thread][script][name] = [];
        }

        if (!(thread in stack)) {
          stack[thread] = {};
        }
        if (!(script in stack[thread])) {
          stack[thread][script] = {};
        }
        if (!(name in stack[thread][script])) {
          stack[thread][script][name] = [];
        }

        var new_duration = true;
        var _length = Object.keys(this.script[thread][script][name]).length;
        if (_length >= 1) {
          for (var period in this.script[thread][script][name]) {
            if (this.script[thread][script][name].hasOwnProperty(period)) {
              if (period.length >= 2 && js_start >= period[0] && js_end <= period[1]) {
                new_duration = false;
                break;
              }
            }
          }
        }
        if (new_duration) {
          this.script[thread][script][name].push([js_start, js_end]);
        }
      }
      var slice_usecs = this.cpu['slice_usecs'];
      var first_slice = parseInt(parseFloat(start) / parseFloat(slice_usecs));
      var last_slice = parseInt(parseFloat(end) / parseFloat(slice_usecs));
      for (var slice_number = first_slice; slice_number < last_slice + 1; slice_number++) {
        var slice_start = slice_number * slice_usecs;
        var slice_end = slice_start + slice_usecs;
        var used_start = Math.max(slice_start, start);
        var used_end = Math.min(slice_end, end);
        var slice_elapsed = used_end - used_start;
        this.adjustTimelineSlice(thread, slice_number, name, parent, slice_elapsed);
      }
      //Recursively process any child events
      if ('c' in timeline_event) {
        for (var child in timeline_event['c']) {
          if (timeline_event['c'].hasOwnProperty(child)) {
            this.processTimelineEvent({ timeline_event: child, parent: name, stack: stack });
          }
        }
      }
    }
  }

  //
  adjustTimelineSlice(thread, slice_number, name, parent, elapsed) {
    try {
      // Don't bother adjusting if both the current event and parent are the same category
      // since they would just cancel each other out.
      if (name !== parent) {
        var fraction = Math.min(1.0, (parseFloat(elapsed) / parseFloat(this.cpu['slice_usecs'])));
        this.cpu['slices'][thread][name][slice_number] += fraction;
        this.cpu['slices'][thread]['total'][slice_number] += fraction;
        if (parent !== null && this.cpu['slices'][thread][parent][slice_number] >= fraction) {
          this.cpu['slices'][thread][parent][slice_number] -= fraction;
          this.cpu['slices'][thread]['total'][slice_number] -= fraction;
        }
        // Make sure we didn't exceed 100% in this slice
        this.cpu['slices'][thread][name][slice_number] =
          Math.min(1.0, this.cpu['slices'][thread][name][slice_number]);
        // make sure we don't exceed 100% for any slot
        if (this.cpu['slices'][thread]['total'][slice_number] > 1.0) {
          var available = Math.max(0.0, 1.0 - fraction);
          for (var slice_name in this.cpu['slices'][thread]) {
            if ((this.cpu['slices'][thread]).hasOwnProperty(slice_name)) {
              if (slice_name !== name) {
                this.cpu['slices'][thread][slice_name][slice_number] =
                  Math.min(this.cpu['slices'][thread][slice_name][slice_number], available);
                available = Math.max(0.0, (available - this.cpu['slices'][thread][slice_name][slice_number]));
              }
            }
          }
          this.cpu['slices'][thread]['total'][slice_number] = Math.min(1.0, Math.max(0.0, 1.0 - available));
        }
      }
    }
    catch (err) {
      console.log(err);
    }
  }
  //
  //ProcessNetworkEvents
  //
  processNetworkEvents(network_trace_events) {
    return new Promise((resolve, reject) => {
      var host = 'tmpURL';
      var _length = network_trace_events.length;
      var ignoreReqId = new Set([]);
      for (var i = 0; i < _length; i++) {
        var net_trace = network_trace_events[i];
        var _request_id = net_trace['args']['data']['requestId'];
        if (!(_request_id in this.network) && !(ignoreReqId.has(_request_id))) {
          this.network[_request_id] = {};
        }
        else if (ignoreReqId.has(_request_id)) {
          continue;
        }
        if (net_trace['name'] === 'ResourceSendRequest') {
          var _url = net_trace['args']['data']['url'];
          if (_url.startsWith('chrome')) {
            ignoreReqId.add(_request_id);
            if (_request_id in this.network) {
              delete this.network[_request_id];
            }
            continue;
          }
          var _startTime = net_trace['ts'];
          if (_url.startsWith("data:")) {
            this.network[_request_id]['url'] = host + "/inlinedObject"; //+ Math.random().toString(36).substring(7);
          }
          else {
            if (_url.includes('#')) {
              this.network[_request_id]['url'] = _url.split('#')[0];
            }
            else {
              this.network[_request_id]['url'] = _url;
            }
          }
          this.network[_request_id]['startTime'] = (_startTime - this.startTime) / 1000;
          if ('stackTrace' in net_trace['args']['data']) {
            var _script_url = net_trace['args']['data']['stackTrace'][0]['url'];
            if (_script_url.includes('#')) {
              _script_url = _script_url.split('#')[0];
            }
            this.network[_request_id]['fromScript'] = _script_url;
          }
          else {
            this.network[_request_id]['fromScript'] = 'Null';
          }
        }
        else if (net_trace['name'] === 'ResourceReceiveResponse' && !(ignoreReqId.has(_request_id))) {
          var _statusCode = net_trace['args']['data']['statusCode'];
          var _mimeType = net_trace['args']['data']['mimeType'];
          var _responseReceivedTime = net_trace['ts'];
          this.network[_request_id]['statusCode'] = _statusCode;
          this.network[_request_id]['responseReceivedTime'] =
            (_responseReceivedTime - this.startTime) / 1000;
          this.network[_request_id]['mimeType'] = _mimeType;
        }
        else if (net_trace['name'] === 'ResourceReceivedData' && !(ignoreReqId.has(_request_id))) {
          var _encodedDataLength = net_trace['args']['data']['encodedDataLength'];
          if (_encodedDataLength !== 0 || !('transferSize' in this.network[_request_id])) {
            this.network[_request_id]['transferSize'] = _encodedDataLength;
          }
        }
        else if (net_trace['name'] === 'ResourceFinish' && !(ignoreReqId.has(_request_id))) {
          var _endTime = net_trace['ts'];
          var _didFail = net_trace['args']['data']['didFail'];
          if (!(_didFail)) {
            this.network[_request_id]['endTime'] = (_endTime - this.startTime) / 1000;
          }
          else {
            delete this.network[_request_id];
          }
        }
        else if (!(ignoreReqId.has(_request_id))) {
          reject('Unknown network name in net_trace');
        }
      }
      resolve();
    });
  }

  //
  // ProcessLoadingEvents
  //
  processLoadingEvents(loading_trace_events) {
    return new Promise((resolve, reject) => {
      const load_list = [];
      let tmpStack = [];
      const _length = loading_trace_events.length;
      for (let i = 0; i < _length; i++) {
        const loading_event = loading_trace_events[i];
        loading_event['ts'] = (loading_event['ts'] - this.startTime) / 1000;
        if (loading_event['ph'] === 'B' || loading_event['ph'] === 'E') {
          if (loading_event['ph'] === 'B') {
            tmpStack.push(loading_event);
          } else {
            if (tmpStack.length > 0) {
              tmpStack.push(loading_event);
            }
            else {
              console.log('E detected without any B');
              continue;
            }
          }
          if (common.isBalanced(tmpStack)) {
            load_list.push(common.mergeEvents(tmpStack));
            tmpStack = [];
          }
        }
        // ParseAuthorStyleSheet
        else if (loading_event['ph'] === 'X') {
          // The ts parameter indicate star ttime of the 'complete (X)' event.
          loading_event['dur'] /= 1000;
          load_list.push([[loading_event]]);
        }
      }
      const _length2 = load_list.length;
      for (let i = 0; i < _length2; i++) {
        this.loading['Loading_' + i.toString()] = {};
        this.loading['Loading_' + i.toString()]['fromScript'] = null;
        this.loading['Loading_' + i.toString()]['styleSheetUrl'] = null;
        this.loading['Loading_' + i.toString()]['url'] = null;
        const _name = load_list[i][0][0]['name'];
        this.loading['Loading_' + i.toString()]['name'] = _name;
        const _startTime = load_list[i][0][0]['ts'];
        this.loading['Loading_' + i.toString()]['startTime'] = _startTime;
        if (load_list[i][0][0]['args']['beginData'] !== undefined) {
          const _pageURL = load_list[i][0][0]['args']['beginData']['url'].split('#')[0];
          this.loading['Loading_' + i.toString()]['url'] = _pageURL;
        }
        if (load_list[i][0][0]['ph'] === 'B') {
          const _endTime = load_list[i][0][1]['ts'];
          this.loading['Loading_' + i.toString()]['endTime'] = _endTime;
          if ('stackTrace' in load_list[i][0][0]['args']['beginData']) {
            let _scriptUrl = load_list[i][0][0]['args']['beginData']['stackTrace'][0]['url'];
            if (_scriptUrl.includes('#')) {
              _scriptUrl = _scriptUrl.split('#')[0];
            }
            this.loading['Loading_' + i.toString()]['fromScript'] = _scriptUrl;
          }
        }
        else if (load_list[i][0][0]['ph'] === 'X') {
          const _duration = load_list[i][0][0]['dur'];
          const _endTime = _startTime + _duration;
          this.loading['Loading_' + i.toString()]['endTime'] = _endTime;
          if ('data' in load_list[i][0][0]['args']) {
            const _styleSheetUrl = load_list[i][0][0]['args']['data']['styleSheetUrl'].split('#')[0];
            this.loading['Loading_' + i.toString()]['styleSheetUrl'] = _styleSheetUrl;
          }
        }
      }
      this.loading = new Map(Object.entries(this.loading));
      this.loading = new Map([...this.loading.entries()].sort((a, b) => parseInt(a[0].split('_')[1]) - parseInt(b[0].split('_')[1])));
      resolve();
    });
  }

  processRenderingEvents(rendering_trace_events) {
    return new Promise((resolve, reject) => {
      var render_list = [];
      var tmpStack = [];
      var _length = rendering_trace_events.length;
      for (var i = 0; i < _length; i++) {
        var render_event = rendering_trace_events[i];
        render_event['ts'] = (render_event['ts'] - this.startTime) / 1000;
        if (render_event['ph'] === 'B' || render_event['ph'] === 'E') {
          if (render_event['ph'] === 'B') {
            tmpStack.push(render_event);
          }
          else if (render_event['ph'] === 'E') {
            if (tmpStack.length > 0) {
              tmpStack.push(render_event);
            }
            else {
              console.log('E detected without any B in: ' + render_event['name']);
              continue;
            }
          }
          if (common.isBalanced(tmpStack)) {
            render_list.push(common.mergeEvents(tmpStack));
            tmpStack = [];
          }
        }
        else if (render_event['ph'] === 'X') {
          if ('dur' in render_event) {
            render_event['dur'] /= 1000;
            render_list.push([[render_event]]);
          }
        }
      }
      var _length2 = render_list.length;
      for (var i = 0; i < _length2; i++) {
        this.rendering['Rendering_' + i.toString()] = {};
        var _startTime = render_list[i][0][0]['ts'];
        this.rendering['Rendering_' + i.toString()]['startTime'] = _startTime;
        var _name = render_list[i][0][0]['name'];
        this.rendering['Rendering_' + i.toString()]['name'] = _name;
        if (render_list[i][0][0]['ph'] === 'B') {
          var _endTime = render_list[i][0][1]['ts'];
          this.rendering['Rendering_' + i.toString()]['endTime'] = _endTime;
        }
        else if (render_list[i][0][0]['ph'] === 'X') {
          var _duration = render_list[i][0][0]['dur'];
          var _endTime = _startTime + _duration;
          this.rendering['Rendering_' + i.toString()]['endTime'] = _endTime;
        }
      }
      this.rendering = new Map(Object.entries(this.rendering));
      this.rendering = new Map([...this.rendering.entries()].sort((a, b) => parseInt(a[0].split('_')[1]) - parseInt(b[0].split('_')[1])));
      resolve();
    });
  }
  ///
  //
  ///
  processPaintingEvents(painting_trace_events) {
    return new Promise((resolve, reject) => {
      var paint_list = [];
      var tmpStack = [];
      var _length = painting_trace_events.length;
      for (var i = 0; i < _length; i++) {
        var paint_event = painting_trace_events[i];
        paint_event['ts'] = (paint_event['ts'] - this.startTime) / 1000;
        if (paint_event['ph'] === 'B' || paint_event['ph'] === 'E') {
          if (paint_event['ph'] === 'B') {
            tmpStack.push(paint_event);
          }
          else if (paint_event['ph'] === 'E') {
            if (tmpStack.length > 0) {
              tmpStack.push(paint_event);
            }
            else {
              console.log('E detected without any B');
              continue;
            }
          }
          if (common.isBalanced(tmpStack)) {
            paint_list.push(common.mergeEvents(tmpStack));
            tmpStack = [];
          }
        }
        else if (paint_event['ph'] === 'X') {
          if ('dur' in paint_event) {
            paint_event['dur'] /= 1000;
          }
          else {
            paint_event['dur'] = 0;
          }
          paint_list.push([[paint_event]]);
        }
      }
      var _length2 = paint_list.length;
      for (var i = 0; i < _length2; i++) {
        this.painting['Painting_' + i.toString()] = {};
        var _startTime = paint_list[i][0][0]['ts'];
        this.painting['Painting_' + i.toString()]['startTime'] = _startTime;
        var _name = paint_list[i][0][0]['name'];
        this.painting['Painting_' + i.toString()]['name'] = _name;
        if (paint_list[i][0][0]['ph'] === 'B') {
          var _endTime = paint_list[i][0][1]['ts'];
          this.painting['Painting_' + i.toString()]['endTime'] = _endTime;
          if ('args' in paint_list[i][0][0]) {
            if ('layerTreeId' in paint_list[i][0][0]['args']) {
              var _layerTreeId = paint_list[i][0][0]['args']['layerTreeId'];
            }
            else {
              var _layerTreeId = null;
            }
            this.painting['Painting_' + i.toString()]['layerTreeId'] = _layerTreeId;
          }
        }
        else if (paint_list[i][0][0]['ph'] === 'X') {
          var _duration = paint_list[i][0][0]['dur'];
          var _endTime = _startTime + _duration;
          this.painting['Painting_' + i.toString()]['endTime'] = _endTime;
        }
      }
      this.painting = new Map(Object.entries(this.painting));
      this.painting = new Map([...this.painting.entries()].sort((a, b) => parseInt(a[0].split('_')[1]) - parseInt(b[0].split('_')[1])));
      resolve();
    });
  }

  //
  //
  sortEventsByTs() {
    return new Promise((resolve, reject) => {
      this.loadingTraceEvents.sort(function (a, b) {
        return a['ts'] - b['ts'];
      })
      this.paintingTraceEvents.sort(function (a, b) {
        return a['ts'] - b['ts'];
      })
      this.renderingTraceEvents.sort(function (a, b) {
        return a['ts'] - b['ts'];
      })
      this.networkTraceEvents.sort(function (a, b) {
        return a['ts'] - b['ts'];
      })
      if (this.netlogTraceEvents.length >= 1) {
        this.netlogTraceEvents.sort(function (a, b) {
          return a['ts'] - b['ts'];
        })
        // Convert the source event id to hex if one exists in netlog
        common.convertIdtoHex(this.netlogTraceEvents);
      }
      resolve();
    });
  }

  sortAllProcessedByStartTime() {
    return new Promise((resolve, reject) => {
      var max_net_time = [['', 0]];
      var max_load_time = [['', 0]];
      var max_script_time = [['', 0]];
      // sort all processed events by startTime
      for (var _id in this.network) {
        if (this.network.hasOwnProperty(_id)) {
          if ('startTime' in this.network[_id] && this.network[_id]['startTime'] >= 0
            && 'endTime' in this.network[_id] && this.network[_id]['endTime'] >= 0) {
            this.networkList.push([_id, this.network[_id]]);
          }
        }
      }
      this.networkList.sort(function (a, b) {
        return a[1]['startTime'] - b[1]['startTime'];
      })

      var _length = this.networkList.length;
      var temp_net_list = [];
      for (var i = 0; i < _length; i++) {
        temp_net_list = ['Networking_' + i.toString(), { 'id': this.networkList[i][0] }];
        temp_net_list[1]['startTime'] = this.networkList[i][1]['startTime'];
        temp_net_list[1]['endTime'] = this.networkList[i][1]['endTime'];
        if (temp_net_list[1]['endTime'] > max_net_time[0][1]) {
          max_net_time[0][1] = temp_net_list[1]['endTime'];
          max_net_time[0][0] = 'Networking_' + i.toString();
        }
        temp_net_list[1]['mimeType'] = this.networkList[i][1]['mimeType'];
        temp_net_list[1]['url'] = this.networkList[i][1]['url'];
        temp_net_list[1]['fromScript'] = this.networkList[i][1]['fromScript'];
        if ('transferSize' in this.networkList[i][1]) {
          temp_net_list[1]['transferSize'] = this.networkList[i][1]['transferSize'];
        }
        temp_net_list[1]['responseReceivedTime'] = this.networkList[i][1]['responseReceivedTime'];
        temp_net_list[1]['statusCode'] = this.networkList[i][1]['statusCode'];
        this.networkList[i] = temp_net_list;
        _url = this.networkList[i][1]['url'];

        if (!(_url in this.networkLookupUrl)) {
          this.networkLookupUrl[_url] = [this.networkList[i][0]];
        }
        else {
          this.networkLookupUrl[_url].push(this.networkList[i][0]);
        }
        this.networkLookupId['Networking_' + i.toString()] = temp_net_list[1];
        temp_net_list = [];
      }
      //Map
      for (let [_id, _load_dict] of this.loading) {
        if ('startTime' in _load_dict && _load_dict['startTime'] >= 0
          && 'endTime' in _load_dict && _load_dict['endTime'] >= 0) {
          if ('url' in _load_dict && _load_dict['url'] !== null) {
            if (_load_dict['url'].startsWith('chrome')) {
              continue;
            }
          }
          if ('fromScript' in _load_dict && _load_dict['fromScript'] !== null) {
            if (_load_dict['fromScript'].startsWith('chrome')) {
              continue;
            }
          }
          this.loadingList.push([_id, _load_dict]);
        }
      }
      var _length = this.loadingList.length;
      for (var i = 0; i < _length; i++) {
        var t_id = this.loadingList[i][0];
        var t_dict = this.loadingList[i][1];
        var _url = t_dict['url'];
        if (!(_url in this.loadingLookupUrl)) {
          this.loadingLookupUrl[_url] = [t_id];
        }
        else {
          this.loadingLookupUrl[_url].push(t_id);
        }
        this.loadingLookupId[t_id] = t_dict;
        if (t_dict['endTime'] > max_load_time[0][1]) {
          max_load_time[0][1] = t_dict['endTime'];
          max_load_time[0][0] = t_id;
        }
      }
      this.loadingList.sort(function (a, b) {
        return a[1]['startTime'] - b[1]['startTime'];
      })
      var _main_thread = this.script['main_thread'];
      var scripts_main_thread = this.script[_main_thread.toString()];
      for (var _id in scripts_main_thread) {
        if (scripts_main_thread.hasOwnProperty(_id)) {
          var script_dict = scripts_main_thread[_id];
          if (_id.startsWith('chrome')) {
            continue;
          }
          if ('EvaluateScript' in script_dict && script_dict['EvaluateScript'].length > 0) {
            var _length2 = script_dict['EvaluateScript'].length;
            for (var j = 0; j < _length2; j++) {
              this.scriptList.push([_id, {
                'startTime': script_dict['EvaluateScript'][j][0],
                'endTime': script_dict['EvaluateScript'][j][1]
              }]);
            }
          }
        }
      }
      for (var _thread in this.script) {
        if (this.script.hasOwnProperty(_thread) && _thread !== 'main_thread' && _thread !== _main_thread) {
          var scripts_non_main_thread = this.script[_thread];
          for (var _id in scripts_non_main_thread) {
            if (scripts_non_main_thread.hasOwnProperty(_id)) {
              var script_dict = scripts_non_main_thread[_id];
              if (_id.startsWith('chrome') || this.scriptList.includes(_id)) {
                continue;
              }
              if ('EvaluateScript' in script_dict && script_dict['EvaluateScript'].length > 0) {
                var _length2 = script_dict['EvaluateScript'].length;
                for (var j = 0; j < _length2; j++) {
                  this.scriptList.push([_id, {
                    'startTime': script_dict['EvaluateScript'][j][0],
                    'endTime': script_dict['EvaluateScript'][j][1]
                  }]);
                }
              }
            }
          }
        }
      }
      var scriptIdsSoFar = common.getCol(this.scriptList, 0);
      for (var _thread in this.scriptXtra) {
        if (this.scriptXtra.hasOwnProperty(_thread)) {
          for (var _id in this.scriptXtra[_thread]) {
            if (this.scriptXtra[_thread].hasOwnProperty(_id)) {
              if (!(scriptIdsSoFar.includes(_id)) && !(_id.startsWith('chrome'))) {
                var script_xtra_dict = this.scriptXtra[_thread][_id];
                if (script_xtra_dict['EvaluateScript'].length > 0) {
                  var _length2 = script_xtra_dict['EvaluateScript'].length;
                  for (var j = 0; j < _length2; j++) {
                    var _startTime = (script_xtra_dict['EvaluateScript'][j][1] - this.startTime) / 1000;
                    var _duration = script_xtra_dict['EvaluateScript'][j][0] / 1000;
                    var _endTime = _startTime + _duration;
                    this.scriptList.push([_id, { 'startTime': _startTime, 'endTime': _endTime }]);
                  }
                }
              }
            }
          }
        }
      }
      this.scriptList.sort(function (a, b) {
        return a[1]['startTime'] - b[1]['startTime'];
      })
      var _listLength = this.scriptList.length;
      for (var i = 0; i < _listLength; i++) {
        this.scriptList[i] = ['Scripting_' + i.toString(),
        {
          'url': this.scriptList[i][0],
          'startTime': this.scriptList[i][1]['startTime'],
          'endTime': this.scriptList[i][1]['endTime']
        }];
        var _url = this.scriptList[i][1]['url'];
        if (!(_url in this.scriptLookupUrl)) {
          this.scriptLookupUrl[_url] = ['Scripting_' + i.toString()];
        }
        else {
          this.scriptLookupUrl[_url].push('Scripting_' + i.toString());
        }
        this.scriptLookupId['Scripting_' + i.toString()] = this.scriptList[i][1];
        if (this.scriptList[i][1]['endTime'] > max_script_time[0][1]) {
          max_script_time[0][1] = this.scriptList[i][1]['endTime'];
          max_script_time[0][0] = 'Scripting_' + i.toString();
        }
      }

      for (let [_id, _render_dict] of this.rendering) {
        if ('startTime' in _render_dict && _render_dict['startTime'] >= 0
          && 'endTime' in _render_dict && _render_dict['endTime'] >= 0) {
          this.renderingList.push([_id, _render_dict]);
        }
      }
      this.renderingList.sort(function (a, b) {
        return a[1]['startTime'] - b[1]['startTime'];
      })


      for (let [_id, _paint_dict] of this.painting) {
        if ('startTime' in _paint_dict && _paint_dict['startTime'] >= 0
          && 'endTime' in _paint_dict && _paint_dict['endTime'] >= 0) {
          this.paintingList.push([_id, _paint_dict]);
        }
      }
      this.paintingList.sort(function (a, b) {
        return a[1]['startTime'] - b[1]['startTime'];
      })

      this.all = this.networkList.concat(this.loadingList, this.scriptList, this.renderingList, this.paintingList);
      this.all.sort(function (a, b) {
        return a[1]['startTime'] - b[1]['startTime'];
      })
      var _allLength = this.all.length;
      for (var k = 0; k < _allLength; k++) {
        this.allStartTimeLookup[this.all[k][0]] = this.all[k][1]['startTime'];
      }

      _allLength = this.all.length;
      for (var l = 0; l < _allLength; l++) {
        this.allDict[this.all[l][0]] = this.all[l][1];
      }
      var _tmp_merged = max_net_time.concat(max_load_time, max_script_time);
      //descending
      _tmp_merged.sort(function (a, b) {
        return b[1] - a[1];
      })
      this.lastActivity = _tmp_merged;

      resolve();
    });
  }
  //
  ///////////////////////////////////////
  //
  edgeStart(e1, s2) {
    if (e1 < s2) {
      return [e1, -1];
    }
    else {
      return [s2, s2];
    }
  }

  findDownload0() {
    var download_0 = null;
    var parse_0 = null;
    var _length = this.all.length;
    for (var i = 0; i < _length; i++) {
      var obj = this.all[i];
      if (obj[0].startsWith('Network')) {
        if (obj[1]['mimeType'] === 'text/html') {
          download_0 = obj;
          // console.log('download_0');
          // console.log(download_0);
          break;
        }
      }
    }
    if (download_0) {
      //var _length = this.all.length;
      for (var j = 0; j < _length; j++) {
        var obj = this.all[j];
        if (obj[0].startsWith('Loading')) {
          if (obj[1]['name'] === 'ParseHTML' &&
            obj[1]['url'] === download_0[1]['url']) {
            parse_0 = obj;
            break;
          }
        }
      }
      if (parse_0) {
        return [download_0, parse_0];
      }
      else {
        console.log('No parse_0');
        return [false, false];
      }
    }
    else {
      return [false, false];
    }
  }

  findIdByUrl(_url, activitiy_data, _type) {
    var activity_startTime = activitiy_data['startTime'];
    var activity_endTime = activitiy_data['endTime'];
    var selected = ['', Infinity];
    if (_type === 'network') {
      var _length = this.networkLookupUrl[_url].length;
      for (var i = 0; i < _length; i++) {
        var net_id = this.networkLookupUrl[_url][i];
        var net_startTime = this.allDict[net_id]['startTime'];
        var net_endTime = this.allDict[net_id]['endTime'];

        if (activity_startTime > net_startTime && activity_endTime > net_endTime) {
          var diff = Math.abs(activity_startTime - net_endTime);
          if (diff < selected[1]) {
            selected = [net_id, diff];
          }
        }
      }
      if (selected[0] === '') {
        return this.download0Id;
      }
      return selected[0];
    }
    else if (_type === 'script') {
      if (_url in this.scriptLookupUrl) {
        var _length = this.scriptLookupUrl[_url].length;

        for (var i = 0; i < _length; i++) {
          var s_id = this.scriptLookupUrl[_url][i];
          // var _startTime = this.scriptList[parseInt(s_id.split('_')[1])][1]['startTime'];
          // var _endTime = this.scriptList[parseInt(s_id.split('_')[1])][1]['endTime'];
          var _startTime = this.allDict[s_id]['startTime'];
          var _endTime = this.allDict[s_id]['endTime'];
          if (activity_startTime > _startTime) {
            var diff = Math.abs(activity_startTime - _endTime);
            if (diff < selected[1]) {
              selected = [s_id, diff];
            }
          }
        }
      }
      return selected[0];
    }
  }
  //JavaScript execution is blocked until CSS is fetched and then CSSOM is built then finish js and build the DOM
  // Find all Scripting's download activities that their download time is before CSS execution
  // --> add dep css_eval--> js_eval
  findBlockingCss(js_networking_endTime, js_scripting_endTime) {
    var id_list = [];
    var length = this.loadingList.length;
    for (var i = 0; i < length; i++) {
      var loadings = this.loadingList[i];
      var load_id = loadings[0];
      var load_data = loadings[1];
      var load_name = load_data['name'];
      if (load_name.startsWith('ParseAuthorStyleSheet')) {
        var _css_endTime = load_data['endTime'];
        if (_css_endTime > js_networking_endTime && _css_endTime < js_scripting_endTime) {
          id_list.push(load_id);
        }
      }
    }
    return id_list;
  }

  // Find latest parse HTML before activity
  findPrevParseId(activity_data) {
    var activity_startTime = activity_data['startTime'];
    var selected = ['', Infinity];
    var _length = this.loadingList.length;
    for (var i = 0; i < _length; i++) {
      var loadings = this.loadingList[i];
      var load_id = loadings[0];
      var load_data = loadings[1];
      var load_name = load_data['name'];
      if (load_name.startsWith('ParseHTML')) {
        var _startTime = load_data['startTime'];
        var _endTime = load_data['endTime'];
        //var _url = load_data['url'];
        // _url not in ['', 'about:blank'] and activity_startTime > _startTime:
        if (activity_startTime > _startTime) {
          var diff = Math.abs(activity_startTime - _endTime);
          if (diff < selected[1]) {
            selected = [load_id, diff];
          }
        }
      }
    }
    if (selected[0] === '' || selected[0] === null) {
      if (activity_startTime > this.allDict[this.parse0Id]['startTime']) {
        return this.parse0Id;
      }
      else if (activity_startTime > this.allDict[this.download0Id]['startTime']) {
        return this.download0Id;
      }
    }

    return selected[0];
  }

  // Find latest scripting before parsing
  findBlockingScripts(activity_data) {
    var activity_startTime = activity_data['startTime'];
    var activity_endTime = activity_data['endTime'];
    var scriptList = [];
    var _length = this.scriptList.length;
    for (var i = 0; i < _length; i++) {
      var scriptings = this.scriptList[i];
      var script_id = scriptings[0];
      var script_data = scriptings[1];
      var _startTime = script_data['startTime'];
      var _endTime = script_data['endTime'];
      //var _url = script_data['url'];
      //if _url not in ['', 'about:blank'] and activity_startTime > _startTime:
      if (activity_startTime < _startTime && activity_endTime > _endTime) {
        scriptList.push(script_id);
      }
    }
    //print('In find_scripting_id', selected[0], activity_data)
    return scriptList;
  }

  findScriptingId(activity_data) {
    var activity_startTime = activity_data['startTime'];
    var selected = ['', Infinity];
    var _length = this.scriptList.length;
    for (var i = 0; i < _length; i++) {
      var scriptings = this.scriptList[i];
      var script_id = scriptings[0];
      var script_data = scriptings[1];
      var _startTime = script_data['startTime'];
      var _endTime = script_data['endTime'];
      //var _url = script_data['url'];
      //if _url not in ['', 'about:blank'] and activity_startTime > _startTime:
      if (activity_startTime > _startTime) {
        var diff = Math.abs(activity_startTime - _endTime);
        if (diff < selected[1]) {
          selected = [script_id, diff];
        }
      }
    }
    if (selected[0] === '' || selected[0] === null) {
      return null;
    }
    //print('In find_scripting_id', selected[0], activity_data)
    return selected[0];
  }

  findScriptingIdBeforeScripting(activity_data) {
    var activity_startTime = activity_data['startTime'];
    var selected = ['', Infinity];
    var _length = this.scriptList.length;
    for (var i = 0; i < _length; i++) {
      var scriptings = this.scriptList[i];
      var script_id = scriptings[0];
      var script_data = scriptings[1];
      var _startTime = script_data['startTime'];
      var _endTime = script_data['endTime'];
      //var _url = script_data['url'];
      //if _url not in ['', 'about:blank'] and activity_startTime > _startTime and activity_startTime > _endTime :
      if (activity_startTime > _startTime && activity_startTime > _endTime) {
        var diff = Math.abs(activity_startTime - _endTime);
        if (diff < selected[1]) {
          selected = [script_id, diff];
        }
      }
    }

    if (selected[0] === '' || selected[0] === null) {
      //print('In find_scripting_id', selected[0], activity_data)
      return null;
    }
    return selected[0];
  }


  extractDependencies() {
    return new Promise((resolve, reject) => {
      var _tmp = this.findDownload0();
      this.download0Id = _tmp[0][0];
      this.parse0Id = _tmp[1][0];
      if (!(this.download0Id && this.parse0Id)) {
        console.log(this.download0Id + '  ' + this.parse0Id);
        reject();
        //return false;
      }
      if (this.allDict[this.download0Id]['startTime'] > this.allDict[this.parse0Id]['startTime']) {
        // console.log(this.download0Id + '  ' +   this.parse0Id);
        reject();
        //return false;
      }

      //Remove loading before parse0ID (devtools' parsing activities)
      var forDeletion = [];
      var _length = this.all.length;

      for (var i = 0; i < _length; i++) {
        var _id = this.all[i][0];
        if (_id.startsWith('Load') &&
          this.allDict[_id]['startTime'] < this.allDict[this.parse0Id]['startTime']) {
          //remove
          forDeletion.push(_id);
        }
      }
      this.all = this.all.filter(item => !forDeletion.includes(item[0]));
      this.loadingList = this.loadingList.filter(item => !forDeletion.includes(item[0]));

      //Initialize this.depsParent
      var _length = this.all.length;
      for (var i = 0; i < _length; i++) {
        var _id = this.all[i][0];
        if (_id.startsWith('Network') || _id.startsWith('Load') ||
          _id.startsWith('Script')) {
          this.depsParent[_id] = []
          this.depsNext[_id] = [];
        }
      }

      _tmp = this.edgeStart(this.allDict[this.download0Id]['endTime'],
        this.allDict[this.parse0Id]['startTime']);
      var a2_startTime = _tmp[0];
      var a1_triggered = _tmp[1];
      this.deps.push({ 'time': a1_triggered, 'a1': this.download0Id, 'a2': this.parse0Id })
      if (a1_triggered === -1) {
        a1_triggered = this.allDict[this.download0Id]['endTime'];
      }
      this.depsParent[this.parse0Id].push([this.download0Id, a1_triggered]);
      this.depsNext[this.download0Id].push(this.parse0Id);
      var _length = this.all.length;

      for (var idx = 0; idx < _length; idx++) {
        var obj = this.all[idx];
        var _nodeId = obj[0];
        var _nodeData = obj[1];
        if (_nodeId.startsWith('Networking') || _nodeId.startsWith('Loading') || _nodeId.startsWith('Scripting')) {
          if (_nodeId.startsWith('Networking')) {
            var _parseID = this.findPrevParseId(_nodeData);
            var _mimeType = _nodeData['mimeType'];
            if (!(this.javascriptTypes.includes(_mimeType))) {
              var _script_nodeId = this.findScriptingId(_nodeData);
            }
            else {
              var _script_nodeId = this.findScriptingIdBeforeScripting(_nodeData);
            }
            //TODO this.allDict[_script_nodeId]['endTime'] > this.allDict[_parseID]['endTime'] misses processLoadingEvents
            if (!(_script_nodeId === null) && this.allDict[_script_nodeId]['endTime'] > this.allDict[_parseID]['endTime']) {
              _tmp = this.edgeStart(this.allDict[_script_nodeId]['endTime'],
                this.allDict[_nodeId]['startTime']);
              var a2_startTime = _tmp[0];
              var a1_triggered = _tmp[1];
              this.deps.push({ 'time': a1_triggered, 'a1': _script_nodeId, 'a2': _nodeId });
              if (a1_triggered === -1) {
                a1_triggered = this.allDict[_script_nodeId]['endTime'];
              }
              this.depsParent[_nodeId].push([_script_nodeId, a1_triggered]);
              this.depsNext[_script_nodeId].push(_nodeId);

            }
            if (_nodeData['startTime'] > this.allDict[this.parse0Id]['startTime']) {
              _tmp = this.edgeStart(this.allDict[_parseID]['endTime'],
                this.allDict[_nodeId]['startTime']);
              var a2_startTime = _tmp[0];
              var a1_triggered = _tmp[1];
              this.deps.push({ 'time': a1_triggered, 'a1': _parseID, 'a2': _nodeId });
              if (a1_triggered === -1) {
                a1_triggered = this.allDict[_parseID]['endTime'];
              }
              this.depsParent[_nodeId].push([_parseID, a1_triggered]);
              this.depsNext[_parseID].push(_nodeId);

            }
            if (!(['Null', null, '', 'null'].includes(_nodeData['fromScript']))) {
              var _tmpurl = _nodeData['fromScript'].split('#')[0];
              try {
                if (_tmpurl in this.scriptLookupUrl) {
                  _script_nodeId = this.findIdByUrl(_tmpurl, _nodeData, 'script');
                }
                else {
                  if (_nodeData['fromScript'].startsWith('https')) {
                    _tmp2 = _tmpurl.replace('https', 'http');
                  }
                  else if (_nodeData['fromScript'].startsWith('http')) {
                    _tmp2 = _tmpurl.replace('http', 'https');
                  }
                  try {
                    if (_tmp2 in this.scriptLookupUrl) {
                      _script_nodeId = this.findIdByUrl(_tmp2, _nodeData, 'network');
                    }
                    else {
                      //inlined
                      _script_nodeId = this.findPrevParseId(_nodeData);
                    }
                  }
                  catch (e) {
                    console.log(e);
                  }
                }
              }
              catch (e) {
                console.log(e);
              }
              //var _script_nodeData = this.scriptList[parseInt(_script_nodeId.split('_')[1])][1];
              if (_script_nodeId) {
                var _script_nodeData = this.allDict[_script_nodeId];
                // There is a js before _nodeId
                if (_script_nodeData['startTime'] < this.allDict[_nodeId]['startTime']) {
                  let _tmp = this.edgeStart(this.allDict[_script_nodeId]['endTime'],
                    this.allDict[_nodeId]['startTime']);
                  var a2_startTime = _tmp[0];
                  var a1_triggered = _tmp[1];
                  this.deps.push({ 'time': a1_triggered, 'a1': _script_nodeId, 'a2': _nodeId });
                  if (a1_triggered === -1) {
                    a1_triggered = this.allDict[_script_nodeId]['endTime'];
                    this.depsParent[_nodeId].push([_script_nodeId, a1_triggered]);
                    this.depsNext[_script_nodeId].push(_nodeId);
                    this.fromScriptSet.add([_script_nodeId, _nodeId]);
                  }
                }
              }
            }
          }
          else if (_nodeId.startsWith('Scripting')) {
            // find related networking for each js eval.
            if (_nodeData['url'] in this.networkLookupUrl) {
              var _network_nodeId = this.findIdByUrl(_nodeData['url'], _nodeData, 'network');
            }
            //var _network_nodeData = this.networkList[parseInt(_network_nodeId.split('_')[1])][1];
            var _network_nodeData = this.allDict[_network_nodeId];
            try {
              _mimeType = _network_nodeData['mimeType'];
            }
            catch (e) {
              console.log('In Scripting catch')
              console.log(_network_nodeData);
              console.log(_network_nodeId);
              console.log(_network_nodeData);
              reject(e);
            }
            var _parseID = this.findPrevParseId(_nodeData);

            if (!(this.javascriptTypes.includes(_mimeType))) {
              _tmp = this.edgeStart(this.allDict[_parseID]['endTime'],
                this.allDict[_nodeId]['startTime']);
              var a2_startTime = _tmp[0];
              var a1_triggered = _tmp[1];
              this.deps.push({ 'time': a1_triggered, 'a1': _parseID, 'a2': _nodeId });
              if (a1_triggered === -1) {
                a1_triggered = this.allDict[_parseID]['endTime'];
              }
              this.depsParent[_nodeId].push([_parseID, a1_triggered]);
              this.depsNext[_parseID].push(_nodeId);

            }
            else if (_network_nodeData['startTime'] < this.allDict[_nodeId]['startTime']) {
              _tmp = this.edgeStart(this.allDict[_network_nodeId]['endTime'],
                this.allDict[_nodeId]['startTime']);
              var a2_startTime = _tmp[0];
              var a1_triggered = _tmp[1];
              this.deps.push({ 'time': a1_triggered, 'a1': _network_nodeId, 'a2': _nodeId });
              if (a1_triggered === -1) {
                a1_triggered = this.allDict[_network_nodeId]['endTime'];
              }
              this.depsParent[_nodeId].push([_network_nodeId, a1_triggered]);
              this.depsNext[_network_nodeId].push(_nodeId);

            }
            /// Add css_eval to js_eval dep
            var _cssEvalIds = this.findBlockingCss(_network_nodeData['endTime'], this.allDict[_nodeId]['endTime']);
            var _csslen = _cssEvalIds.length;
            for (var i = 0; i < _csslen; i++) {
              var _cssEvalId = _cssEvalIds[i];

              /*
              edgeStart(e1, s2){
                if (e1 < s2){
                  return [e1, -1];
                }
                else{
                  return [s2, s2];
                }
              }
              */
              if (_cssEvalId) {
                _tmp = this.edgeStart(this.allDict[_cssEvalId]['endTime'],
                  this.allDict[_nodeId]['startTime']);
                var a2_startTime = _tmp[0];
                var a1_triggered = _tmp[1];
                if (this.allDict[_cssEvalId]['startTime'] > this.allDict[_nodeId]['startTime']) {
                  a1_triggered = this.allDict[_cssEvalId]['startTime'];
                }
                this.deps.push({ 'time': a1_triggered, 'a1': _cssEvalId, 'a2': _nodeId });
                if (a1_triggered === -1) {
                  a1_triggered = this.allDict[_cssEvalId]['endTime'];
                }

                this.depsParent[_nodeId].push([_cssEvalId, a1_triggered]);
                this.depsNext[_cssEvalId].push(_nodeId);

              }
            }
            /// Find scripting before js_eval which finishes after it's download but before js_eval June2018
            var _script_nodeId = this.findScriptingIdBeforeScripting(_nodeData);

            if (!(_script_nodeId === null) && this.allDict[_script_nodeId]['endTime'] > this.allDict[_network_nodeId]['endTime']) {
              _tmp = this.edgeStart(this.allDict[_script_nodeId]['endTime'],
                this.allDict[_nodeId]['startTime']);
              var a2_startTime = _tmp[0];
              var a1_triggered = _tmp[1];
              this.deps.push({ 'time': a1_triggered, 'a1': _script_nodeId, 'a2': _nodeId });
              if (a1_triggered === -1) {
                a1_triggered = this.allDict[_script_nodeId]['endTime'];
              }
              this.depsParent[_nodeId].push([_script_nodeId, a1_triggered]);
              this.depsNext[_script_nodeId].push(_nodeId);

            }
          }
          else if (_nodeId.startsWith('Loading')) {
            if (_nodeData['name'] === 'ParseAuthorStyleSheet') {
              var _tmpurl = _nodeData['styleSheetUrl'].split('#')[0];
              try {
                if (_tmpurl in this.networkLookupUrl) {
                  var _network_nodeId = this.findIdByUrl(_tmpurl, _nodeData, 'network');
                }
                else {
                  if (_nodeData['styleSheetUrl'].startsWith('https')) {
                    var _tmp2 = _nodeData['styleSheetUrl'].replace('https', 'http');
                  }
                  else if (_nodeData['styleSheetUrl'].startsWith('http')) {
                    var _tmp2 = _nodeData['styleSheetUrl'].replace('http', 'https');
                  }
                  try {
                    if (_tmp2 in this.networkLookupUrl) {
                      _network_nodeId = this.findIdByUrl(_tmp2, _nodeData, 'network');
                    }
                  }
                  catch (e) {
                    console.log(e);
                  }
                }
              }
              catch (e) {
                console.log(e);
              }
              var _network_nodeData = this.allDict[_network_nodeId];
              if (_network_nodeData['startTime'] < this.allDict[_nodeId]['startTime']) {
                _tmp = this.edgeStart(this.allDict[_network_nodeId]['endTime'],
                  this.allDict[_nodeId]['startTime']);
                var a2_startTime = _tmp[0];
                var a1_triggered = _tmp[1];
                this.deps.push({ 'time': a1_triggered, 'a1': _network_nodeId, 'a2': _nodeId });
                if (a1_triggered === -1) {
                  a1_triggered = this.allDict[_network_nodeId]['endTime'];
                }
                this.depsParent[_nodeId].push([_network_nodeId, a1_triggered]);
                this.depsNext[_network_nodeId].push(_nodeId);

                ///
                // find latest scripting too
                ///
                var _script_nodeId = this.findScriptingId(_nodeData);
                if (!(_script_nodeId === null)) {
                  _tmp = this.edgeStart(this.allDict[_script_nodeId]['endTime'],
                    this.allDict[_nodeId]['startTime']);
                  var a2_startTime = _tmp[0];
                  var a1_triggered = _tmp[1];
                  this.deps.push({ 'time': a1_triggered, 'a1': _network_nodeId, 'a2': _nodeId });
                  this.fromScriptSet.add((_script_nodeId, _nodeId));
                  if (a1_triggered === -1) {
                    a1_triggered = this.allDict[_script_nodeId]['endTime'];
                  }
                  this.depsParent[_nodeId].push([_script_nodeId, a1_triggered]);
                  this.depsNext[_script_nodeId].push(_nodeId);
                }
                ///
                // Find latest parsHTML too
                ///
                _parseID = this.findPrevParseId(_nodeData);
                _tmp = this.edgeStart(this.allDict[_parseID]['endTime'],
                  this.allDict[_nodeId]['startTime']);
                var a2_startTime = _tmp[0];
                var a1_triggered = _tmp[1];

                this.deps.push({ 'time': a1_triggered, 'a1': _parseID, 'a2': _nodeId });
                if (a1_triggered === -1) {
                  a1_triggered = this.allDict[_parseID]['endTime'];
                }
                this.depsParent[_nodeId].push([_parseID, a1_triggered]);
                this.depsNext[_parseID].push(_nodeId);

              }
            }
            else if (_nodeData['name'] === 'ParseHTML' &&
              ['Null', null, '', 'null'].includes(_nodeData['fromScript'])) {
              if (_nodeData['startTime'] > this.allDict[this.parse0Id]['startTime'] && _nodeData['url'] !== '') {
                if (_nodeData['url'] in this.networkLookupUrl) {
                  if (this.networkLookupUrl[_nodeData['url']].length <= 1 &&
                    this.allDict[this.networkLookupUrl[_nodeData['url']][0]]['startTime'] < _nodeData['startTime']) {
                    var _network_nodeId = this.networkLookupUrl[_nodeData['url']][0];
                  }
                  else {
                    var _network_nodeId = this.findIdByUrl(_nodeData['url'], _nodeData, 'network');

                  }
                }
                var _network_nodeData = this.allDict[_network_nodeId];
                try {
                  _tmp = this.edgeStart(this.allDict[_network_nodeId]['endTime'],
                    this.allDict[_nodeId]['startTime']);
                }
                catch (e) {
                  console.log('_nodeData');
                  console.log(_nodeData);
                  console.log('_network_nodeId' + _network_nodeId);
                  reject(e + 'error!!!');
                }
                var a2_startTime = _tmp[0];
                var a1_triggered = _tmp[1];
                this.deps.push({ 'time': a1_triggered, 'a1': _network_nodeId, 'a2': _nodeId })
                if (a1_triggered === -1) {
                  a1_triggered = this.allDict[_network_nodeId]['endTime'];
                }
                this.depsParent[_nodeId].push([_network_nodeId, a1_triggered]);
                this.depsNext[_network_nodeId].push(_nodeId);

                ///
                // find latest scripting too
                ///
                var _script_nodeId = this.findScriptingId(_nodeData);
                if (!(_script_nodeId === null)) {
                  _tmp = this.edgeStart(this.allDict[_script_nodeId]['endTime'],
                    this.allDict[_nodeId]['startTime']);
                  var a2_startTime = _tmp[0];
                  var a1_triggered = _tmp[1];
                  this.deps.push({ 'time': a1_triggered, 'a1': _script_nodeId, 'a2': _nodeId });
                  this.fromScriptSet.add([_script_nodeId, _nodeId]);

                  if (a1_triggered === -1) {
                    a1_triggered = this.allDict[_script_nodeId]['endTime'];
                  }
                  this.depsParent[_nodeId].push([_script_nodeId, a1_triggered]);
                  this.depsNext[_script_nodeId].push(_nodeId);

                }
                ///
                // Find latest parsHTML too
                ///
                var _parseID = this.findPrevParseId(_nodeData);
                _tmp = this.edgeStart(this.allDict[_parseID]['endTime'],
                  this.allDict[_nodeId]['startTime']);
                var a2_startTime = _tmp[0];
                var a1_triggered = _tmp[1];
                this.deps.push({ 'time': a1_triggered, 'a1': _parseID, 'a2': _nodeId });
                if (a1_triggered === -1) {
                  a1_triggered = this.allDict[_parseID]['endTime'];
                }
                this.depsParent[_nodeId].push([_parseID, a1_triggered]);
                this.depsNext[_parseID].push(_nodeId);

                //findBlockingScripts
                var blockingScripts = this.findBlockingScripts(_nodeData);
                for (var i = 0; i < blockingScripts.length; i++) {
                  var _bscriptId = blockingScripts[i];
                  var a2_startTime = this.allDict[_bscriptId]['startTime'];
                  var a1_triggered = this.allDict[_bscriptId]['startTime'];
                  this.deps.push({ 'time': a1_triggered, 'a1': _nodeId, 'a2': _bscriptId });
                  this.depsParent[_bscriptId].push([_nodeId, a1_triggered]);
                  var a2_startTime = this.allDict[_bscriptId]['endTime'];
                  var a1_triggered = this.allDict[_bscriptId]['endTime'];
                  this.deps.push({ 'time': a1_triggered, 'a1': _bscriptId, 'a2': _nodeId });
                  this.depsParent[_nodeId].push([_bscriptId, a1_triggered]);
                  //this.depsNext[_parseID].push(_nodeId);

                }
              }
              else if (_nodeData['startTime'] > this.allDict[this.parse0Id]['startTime']) {
                // find latest scripting too
                var _script_nodeId = this.findScriptingId(_nodeData);
                if (!(_script_nodeId === null)) {
                  var _tmp = this.edgeStart(this.allDict[_script_nodeId]['endTime'],
                    this.allDict[_nodeId]['startTime']);
                  var a2_startTime = _tmp[0];
                  var a1_triggered = _tmp[1];
                  this.deps.push({ 'time': a1_triggered, 'a1': _script_nodeId, 'a2': _nodeId });
                  this.fromScriptSet.add([_script_nodeId, _nodeId]);

                  if (a1_triggered === -1) {
                    a1_triggered = this.allDict[_script_nodeId]['endTime'];
                  }
                  this.depsParent[_nodeId].push([_script_nodeId, a1_triggered]);
                  this.depsNext[_script_nodeId].push(_nodeId);
                }

                ///
                // Find latest parsHTML too
                ///
                var _parseID = this.findPrevParseId(_nodeData);

                var _tmp = this.edgeStart(this.allDict[_parseID]['endTime'],
                  this.allDict[_nodeId]['startTime']);
                var a2_startTime = _tmp[0];
                var a1_triggered = _tmp[1];
                this.deps.push({ 'time': a1_triggered, 'a1': _parseID, 'a2': _nodeId });
                if (a1_triggered === -1) {
                  a1_triggered = this.allDict[_parseID]['endTime'];
                }
                this.depsParent[_nodeId].push([_parseID, a1_triggered]);
                this.depsNext[_parseID].push(_nodeId);
              }
            }
            else if (_nodeData['name'] === 'ParseHTML' && !(['Null', null, '', 'null'].includes(_nodeData['fromScript']))) {
              // _script_nodeId = this.scriptLookupUrl[urldefrag(_nodeData['fromScript'])[0]]
              var _tmpurl = _nodeData['fromScript'].split('#')[0];
              if (_tmpurl in this.scriptLookupUrl) {
                var _script_nodeId = this.findIdByUrl(_tmpurl, _nodeData, 'script');
              }
              //var _script_nodeData = this.scriptList[parseInt(_script_nodeId.split('_')[1])][1];
              if (_script_nodeId) {
                var _script_nodeData = this.allDict[_script_nodeId];
                // There is a js before _nodeId
                if (_script_nodeData['startTime'] < this.allDict[_nodeId]['startTime']) {
                  _tmp = this.edgeStart(this.allDict[_script_nodeId]['endTime'],
                    this.allDict[_nodeId]['startTime']);
                  var a2_startTime = _tmp[0];
                  var a1_triggered = _tmp[1];
                  this.deps.push({ 'time': a1_triggered, 'a1': _script_nodeId, 'a2': _nodeId });
                  if (a1_triggered === -1) {
                    a1_triggered = this.allDict[_script_nodeId]['endTime'];
                  }
                  this.depsParent[_nodeId].push([_script_nodeId, a1_triggered]);
                  this.depsNext[_script_nodeId].push(_nodeId);
                }
              }

              ///
              // find latest scripting too
              ///
              var _middle_script_nodeId = this.findScriptingIdBeforeScripting(_nodeData);
              if (!(_middle_script_nodeId === null)) {
                _tmp = this.edgeStart(this.allDict[_middle_script_nodeId]['endTime'],
                  this.allDict[_nodeId]['startTime']);
                var a2_startTime = _tmp[0];
                var a1_triggered = _tmp[1];
                this.deps.push({ 'time': a1_triggered, 'a1': _middle_script_nodeId, 'a2': _nodeId });
                if (a1_triggered === -1) {
                  a1_triggered = this.allDict[_middle_script_nodeId]['endTime'];
                }
                this.depsParent[_nodeId].push([_middle_script_nodeId, a1_triggered]);
                this.depsNext[_middle_script_nodeId].push(_nodeId);
              }
              ///
              // Find latest parsHTML too
              ///
              var _parseID = this.findPrevParseId(_nodeData);
              _tmp = this.edgeStart(this.allDict[_parseID]['endTime'],
                this.allDict[_nodeId]['startTime']);
              var a2_startTime = _tmp[0];
              var a1_triggered = _tmp[1];
              this.deps.push({ 'time': a1_triggered, 'a1': _parseID, 'a2': _nodeId });
              if (a1_triggered === -1) {
                a1_triggered = this.allDict[_parseID]['endTime'];
              }
              this.depsParent[_nodeId].push([_parseID, a1_triggered]);
              this.depsNext[_parseID].push(_nodeId);

            }
          }
        }
      }
      //return true;
      resolve();
    });
  }

  findCpMaxEnd(_array, _tailTime) {
    var _max = -1000;
    var _max_id = null;
    var _length = _array.length;
    for (var i = 0; i < _length; i++) {
      var _arr = _array[i];
      var parentId = _arr[0];
      var parentTime = _arr[1];
      if (parentTime > _max && parentTime < _tailTime) {
        _max = parentTime;
        _max_id = parentId;
      }
    }
    return [_max_id, _max];
  }


  findCriticalPath(source) {
    return new Promise((resolve, reject) => {
      var tailTime = this.allDict[source]['endTime'];
      var _result = [];
      var parent = source;
      var i = 0;
      this.criticalPath.push(source);
      while (parent !== this.download0Id && i < 1000) {
        i += 1;
        if (parent in this.depsParent) {
          _result = this.findCpMaxEnd(this.depsParent[parent], tailTime);
          parent = _result[0];
          tailTime = _result[1];
          this.criticalPath.push(parent);
        }
      }
      resolve();
    });
  }

  findCriticalPathOld(source, tailTime) {
    return new Promise((resolve, reject) => {
      try {
        this.criticalPath = this.criticalPath.concat([source]);
        if (source in this.depsParent) {
          var _result = this.findCpMaxEnd(this.depsParent[source], tailTime);
          var source = _result[0];
          tailTime = _result[1];
          return this.findCriticalPath(source, tailTime);
        }
        else {
          return;
        }
      }
      catch (e) {
        console.log(source);
        console.log(this.depsParent[source]);
        reject(e);
        //throw(e);
      }
      resolve();
    });

  }

  ///
  orderLayout() {
    return new Promise((resolve, reject) => {
      var i = 0;
      var _length = this.networkList.length;
      for (var idx = 0; idx < _length; idx++) {
        var net_obj = this.networkList[idx];
        var _url = net_obj[1]['url'];
        if (!(_url in this.ordered)) {
          this.ordered[_url] = [net_obj[0]];
          this.orderedUrlLookup[_url] = i;
          i += 1;
        }
        else {
          this.ordered[_url].push(net_obj[0]);
        }
      }
      var _length2 = this.scriptList.length;
      for (var idx2 = 0; idx2 < _length2; idx2++) {
        var script_obj = this.scriptList[idx2];
        var _url = script_obj[1]['url'];
        if (!(_url in this.ordered)) {
          this.ordered[_url] = [script_obj[0]];
          this.orderedUrlLookup[_url] = i;
          i += 1;
        }
        else {
          this.ordered[_url].push(script_obj[0]);
        }
      }

      var _length3 = this.loadingList.length;
      for (var idx3 = 0; idx3 < _length3; idx3++) {
        var load_obj = this.loadingList[idx3];
        if (load_obj[1]['name'] === 'ParseHTML') {
          var _url = load_obj[1]['url'];
        }
        else if (load_obj[1]['name'] === 'ParseAuthorStyleSheet') {
          var _url = load_obj[1]['styleSheetUrl'];
        }
        if (!(_url in this.ordered)) {
          this.ordered[_url] = [load_obj[0]];
          this.orderedUrlLookup[_url] = i;
          i += 1;
        }
        else {
          this.ordered[_url].push(load_obj[0]);
        }
      }
      this.ordered = Object.entries(this.ordered);
      var that = this;
      this.ordered = this.ordered.sort(function (a, b) {
        return that.allStartTimeLookup[a[1][0]] - that.allStartTimeLookup[b[1][0]];
      })
      resolve();
    });
  }
  ///
  ///

  writeOutputlog() {
    var _length = this.ordered.length;
    for (var i = 0; i < _length; i++) {
      var _value = this.ordered[i];
      var _url_group = _value[0];
      var _node_Id_list = _value[1];
      var _tmp_list = [];
      var _length2 = _node_Id_list.length;
      for (var j = 0; j < _length2; j++) {
        var _nodeId = _node_Id_list[j];
        var _tmp_dict = {};
        var _tmp_merged_dict = {};
        if (_nodeId.startsWith('Network')) {
          _tmp_dict['activityId'] = _nodeId;
          _tmp_merged_dict = Object.assign(_tmp_dict, this.networkLookupId[_nodeId]);
          _tmp_list.push(_tmp_merged_dict);
        }
        else if (_nodeId.startsWith('Load')) {
          _tmp_dict['activityId'] = _nodeId;
          _tmp_merged_dict = Object.assign(_tmp_dict, this.loadingLookupId[_nodeId]);
          _tmp_list.push(_tmp_merged_dict);
        }
        else if (_nodeId.startsWith('Script')) {
          _tmp_dict['activityId'] = _nodeId;
          _tmp_merged_dict = Object.assign(_tmp_dict, this.scriptLookupId[_nodeId]);
          _tmp_list.push(_tmp_merged_dict);
        }
      }
      this.output.push({ 'id': _url_group, 'objs': _tmp_list });
    }
    
    const lists = {
      networking: this.networkList,
      loading: this.loadingList,
      scripting: this.scriptList,
      rendering: this.renderingList,
      painting: this.paintingList
    };

    const dicts = {
      networking: {},
      loading: {},
      scripting: {},
      rendering: {},
      painting: {}
    };

    Object.entries(lists).map(([key, value]) => {
      for (var i = 0; i < value.length; i++) {
        const _tmp_dict = {};
        const _value = value[i];
        const _nodeId = _value[0];
        _tmp_dict['activityId'] = _nodeId;
        const _nodeData = _value[1];
        const _tmp_merged_dict = Object.assign(_tmp_dict, _nodeData);
        dicts[key][_nodeId] = _tmp_merged_dict;
      }
    });

    const critical_path = this.criticalPath.map((activity) => {
      if (activity.includes('Networking')) {
        return dicts['networking'][activity];
      }
      if (activity.includes('Loading')) {
        return dicts['loading'][activity];
      }
      if (activity.includes('Scripting')) {
        return dicts['scripting'][activity];
      }
      if (activity.includes('Rendering')) {
        return dicts['rendering'][activity];
      }
      return dicts['painting'][activity];
    });

    const output = {
      'criticalPath': critical_path,
      'networkingTimeCp': this.networkingCp,
      'loadingTimeCp': this.loadingCp,
      'scriptingTimeCp': this.scriptingCp,
      'firstMeaningfulPaint': this.firstMeaningfulPaint,
      'firstContentfulPaint': this.firstContentfulPaint,
      'firstPaint': this.firstPaint,
      'loadEventEnd': this.loadEventEnd,
      'domContentLoadedEventEnd': this.domContentLoadedEventEnd,
      'timeToInteractive': this.timeToInteractive,
      'loading': dicts['loading'],
      'rendering': dicts['rendering'],
      'painting': dicts['painting']
    };

    return output;
  }


  findInterestingTimes() {
    return new Promise((resolve, reject) => {
      if (this.mainFrameId in this.firstMeaningfulPaintDict) {
        if (this.firstMeaningfulPaintDict.hasOwnProperty(this.mainFrameId)) {
          this.firstMeaningfulPaintList = this.firstMeaningfulPaintDict[this.mainFrameId];
        }
      }

      if (this.mainFrameId in this.firstMeaningfulPaintCandidatesDict) {
        if (this.firstMeaningfulPaintCandidatesDict.hasOwnProperty(this.mainFrameId)) {
          this.firstMeaningfulPaintCandidatesList = this.firstMeaningfulPaintCandidatesDict[this.mainFrameId];
        }
      }

      if (this.mainFrameId in this.firstPaintDict) {
        if (this.firstPaintDict.hasOwnProperty(this.mainFrameId)) {
          this.firstPaintList = this.firstPaintDict[this.mainFrameId];
        }
      }

      if (this.mainFrameId in this.firstContentfulPaintDict) {
        if (this.firstContentfulPaintDict.hasOwnProperty(this.mainFrameId)) {
          this.firstContentfulPaintList = this.firstContentfulPaintDict[this.mainFrameId];
        }
      }

      if (this.mainFrameId in this.loadEventEndDict) {
        if (this.loadEventEndDict.hasOwnProperty(this.mainFrameId)) {
          this.loadEventEndList = this.loadEventEndDict[this.mainFrameId];
        }
      }
      if (this.mainFrameId in this.domContentLoadedEventEndDict) {
        if (this.domContentLoadedEventEndDict.hasOwnProperty(this.mainFrameId)) {
          this.domContentLoadedEventEndList = this.domContentLoadedEventEndDict[this.mainFrameId];
        }
      }
      if (this.firstMeaningfulPaintList.length < 1) {
        this.firstMeaningfulPaintList = this.firstMeaningfulPaintCandidatesList;
      }

      if (this.firstMeaningfulPaintList.length > 0) {
        this.firstMeaningfulPaint = this.firstMeaningfulPaintList.sort(function (a, b) { return a - b });
        this.firstMeaningfulPaint = (this.firstMeaningfulPaint[this.firstMeaningfulPaint.length - 1] - this.startTime) / 1000;
      }

      if (this.firstContentfulPaintList.length > 0) {
        this.firstContentfulPaint = this.firstContentfulPaintList.sort(function (a, b) { return a - b });
        this.firstContentfulPaint = (this.firstContentfulPaint[this.firstContentfulPaint.length - 1] - this.startTime) / 1000;
      }
      if (this.firstPaintList.length > 0) {
        this.firstPaint = this.firstPaintList.sort(function (a, b) { return a - b });
        this.firstPaint = (this.firstPaint[this.firstPaint.length - 1] - this.startTime) / 1000;
      }
      if (this.loadEventEndList.length > 0) {
        this.loadEventEnd = this.loadEventEndList.sort(function (a, b) { return a - b });
        this.loadEventEnd = (this.loadEventEnd[this.loadEventEnd.length - 1] - this.startTime) / 1000;
      }
      if (this.domContentLoadedEventEndList.length > 0) {
        this.domContentLoadedEventEnd = this.domContentLoadedEventEndList.sort(function (a, b) { return a - b });
        this.domContentLoadedEventEnd = (this.domContentLoadedEventEnd[this.domContentLoadedEventEnd.length - 1] - this.startTime) / 1000;
      }

      if (this.interactive.length > 0) {
        this.interactive = this.interactive.sort(function (a, b) {
          return (a[0] - b[0]);
        })

        this.timeToInteractive = parseFloat(this.interactive[this.interactive.length - 1][0]);
      }

      resolve();
    });
  }

  timeSummary() {
    return new Promise((resolve, reject) => {
      var criticalPath = this.criticalPath;
      var cpLength = criticalPath.length;
      var i = cpLength - 1;
      var nodeId = null;
      var prev = null;
      var duration = 0.0;
      var endTime = 0.0;
      while (i >= 1) {
        nodeId = criticalPath[i];
        prev = criticalPath[i - 1];
        if (i === cpLength - 1) {
          duration = parseFloat(this.allDict[nodeId]['endTime']) - parseFloat(this.allDict[nodeId]['startTime']);
        }
        else {
          duration = parseFloat(endTime) - parseFloat(this.allDict[nodeId]['startTime']);

        }
        // Calculate triggerTime for prev

        var arrLength = this.depsParent[nodeId].length;
        for (var j = 0; j < arrLength; j++) {
          var arr = this.depsParent[nodeId][j];
          if (arr[0] === prev) {
            endTime = parseFloat(arr[1]);
            break;
          }
        }
        i -= 1;
        if (nodeId.startsWith('Networking')) {
          this.networkingCp = (this.networkingCp) + duration;
        }
        else if (nodeId.startsWith('Loading')) {
          this.loadingCp = (this.loadingCp) + duration;

        }
        else if (nodeId.startsWith('Scripting')) {
          this.scriptingCp = (this.scriptingCp) + duration;
        }
      }

      duration = endTime - parseFloat(this.allDict[prev]['startTime']);

      if (nodeId.startsWith('Networking')) {
        this.networkingCp += duration;
      }
      else if (nodeId.startsWith('Loading')) {
        this.loadingCp += duration;
      }
      else if (nodeId.startsWith('Scripting')) {
        this.scriptingCp += duration;
      }
      this.networkingCp = common.round(this.networkingCp, 2);
      this.loadingCp = common.round(this.loadingCp, 2);
      this.scriptingCp = common.round(this.scriptingCp, 2);
      resolve();
    });
  }
  //class ends
}
module.exports = Analyze;
