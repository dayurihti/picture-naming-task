#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.0),
    on January 09, 2026, at 17:19
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.0'
expName = 'PICTURE_NAMING_TASK'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '"001"',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\college\\psychopy\\PICTURE_NAMING_TASK\\PICTURE_NAMING_TASK.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[1,1,1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1,1,1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instruction" ---
    text_instruction = visual.TextStim(win=win, name='text_instruction',
        text='Anda akan melihat gambar pada setiap trial.\nSebutkan nama gambar tersebut secara lisan, secepat dan seakurat mungkin.\n\nJika muncul bendera Indonesia, jawab dengan Bahasa Indonesia.\nJika muncul bendera Inggris, jawab dengan Bahasa Inggris.\n\n\nKlik di mana saja untuk melanjutkan.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "practice_block" ---
    text_practice = visual.TextStim(win=win, name='text_practice',
        text='Blok pertama adalah blok latihan.\nINGAT: jawab secepat dan seakurat mungkin.\n\n\nKlik di mana saja untuk melanjutkan',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_2.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "fixation_dot_IN" ---
    dot_IN = visual.ShapeStim(
        win=win, name='dot_IN',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "pic_stim_IN" ---
    language_cue_IN = visual.ImageStim(
        win=win,
        name='language_cue_IN', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.3), draggable=False, size=(0.15, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    picture_stimuli_IN = visual.ImageStim(
        win=win,
        name='picture_stimuli_IN', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.04), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "blank_screen_IN" ---
    blank_IN = visual.ImageStim(
        win=win,
        name='blank_IN', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "fixation_dot_EN" ---
    dot_EN = visual.ShapeStim(
        win=win, name='dot_EN',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=(-1.0000, -1.0000, -1.0000),
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "pic_stim_EN" ---
    language_cue_EN = visual.ImageStim(
        win=win,
        name='language_cue_EN', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.3), draggable=False, size=(0.15, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    picture_stimuli_EN = visual.ImageStim(
        win=win,
        name='picture_stimuli_EN', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.04), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "blank_screen_EN" ---
    blank_EN = visual.ImageStim(
        win=win,
        name='blank_EN', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "fixation_dot" ---
    dot = visual.ShapeStim(
        win=win, name='dot',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "pic_stim_lang_cue" ---
    language_cue = visual.ImageStim(
        win=win,
        name='language_cue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.3), draggable=False, size=(0.15, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    picture_stimuli = visual.ImageStim(
        win=win,
        name='picture_stimuli', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.04), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "blank_screen_2" ---
    blank = visual.ImageStim(
        win=win,
        name='blank', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "non_switching_block" ---
    text_non_switching = visual.TextStim(win=win, name='text_non_switching',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instruction" ---
    # create an object to store info about Routine instruction
    instruction = data.Routine(
        name='instruction',
        components=[text_instruction, mouse],
    )
    instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse
    gotValidClick = False  # until a click is received
    # store start times for instruction
    instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction.tStart = globalClock.getTime(format='float')
    instruction.status = STARTED
    thisExp.addData('instruction.started', instruction.tStart)
    instruction.maxDuration = None
    # keep track of which components have finished
    instructionComponents = instruction.components
    for thisComponent in instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruction" ---
    thisExp.currentRoutine = instruction
    instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruction* updates
        
        # if text_instruction is starting this frame...
        if text_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction.frameNStart = frameN  # exact frame index
            text_instruction.tStart = t  # local t and not account for scr refresh
            text_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction.started')
            # update status
            text_instruction.status = STARTED
            text_instruction.setAutoDraw(True)
        
        # if text_instruction is active this frame...
        if text_instruction.status == STARTED:
            # update params
            pass
        # *mouse* updates
        
        # if mouse is starting this frame...
        if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse.frameNStart = frameN  # exact frame index
            mouse.tStart = t  # local t and not account for scr refresh
            mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse.status = STARTED
            mouse.mouseClock.reset()
            prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
        if mouse.status == STARTED:  # only update if started and not finished!
            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    continueRoutine = False  # end routine on response        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instruction,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instruction.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instruction.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction" ---
    for thisComponent in instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction
    instruction.tStop = globalClock.getTime(format='float')
    instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction.stopped', instruction.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.nextEntry()
    # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice_block" ---
    # create an object to store info about Routine practice_block
    practice_block = data.Routine(
        name='practice_block',
        components=[text_practice, mouse_2],
    )
    practice_block.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_2
    gotValidClick = False  # until a click is received
    # store start times for practice_block
    practice_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    practice_block.tStart = globalClock.getTime(format='float')
    practice_block.status = STARTED
    thisExp.addData('practice_block.started', practice_block.tStart)
    practice_block.maxDuration = None
    # keep track of which components have finished
    practice_blockComponents = practice_block.components
    for thisComponent in practice_block.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "practice_block" ---
    thisExp.currentRoutine = practice_block
    practice_block.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_practice* updates
        
        # if text_practice is starting this frame...
        if text_practice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_practice.frameNStart = frameN  # exact frame index
            text_practice.tStart = t  # local t and not account for scr refresh
            text_practice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_practice, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_practice.started')
            # update status
            text_practice.status = STARTED
            text_practice.setAutoDraw(True)
        
        # if text_practice is active this frame...
        if text_practice.status == STARTED:
            # update params
            pass
        # *mouse_2* updates
        
        # if mouse_2 is starting this frame...
        if mouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_2.frameNStart = frameN  # exact frame index
            mouse_2.tStart = t  # local t and not account for scr refresh
            mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_2.status = STARTED
            mouse_2.mouseClock.reset()
            prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
        if mouse_2.status == STARTED:  # only update if started and not finished!
            buttons = mouse_2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    continueRoutine = False  # end routine on response        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=practice_block,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            practice_block.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if practice_block.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in practice_block.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_block" ---
    for thisComponent in practice_block.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for practice_block
    practice_block.tStop = globalClock.getTime(format='float')
    practice_block.tStopRefresh = tThisFlipGlobal
    thisExp.addData('practice_block.stopped', practice_block.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.nextEntry()
    # the Routine "practice_block" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    IN_block = data.TrialHandler2(
        name='IN_block',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('condition_IN.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(IN_block)  # add the loop to the experiment
    thisIN_block = IN_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIN_block.rgb)
    if thisIN_block != None:
        for paramName in thisIN_block:
            globals()[paramName] = thisIN_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisIN_block in IN_block:
        IN_block.status = STARTED
        if hasattr(thisIN_block, 'status'):
            thisIN_block.status = STARTED
        currentLoop = IN_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisIN_block.rgb)
        if thisIN_block != None:
            for paramName in thisIN_block:
                globals()[paramName] = thisIN_block[paramName]
        
        # --- Prepare to start Routine "fixation_dot_IN" ---
        # create an object to store info about Routine fixation_dot_IN
        fixation_dot_IN = data.Routine(
            name='fixation_dot_IN',
            components=[dot_IN],
        )
        fixation_dot_IN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation_dot_IN
        fixation_dot_IN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_dot_IN.tStart = globalClock.getTime(format='float')
        fixation_dot_IN.status = STARTED
        thisExp.addData('fixation_dot_IN.started', fixation_dot_IN.tStart)
        fixation_dot_IN.maxDuration = None
        # keep track of which components have finished
        fixation_dot_INComponents = fixation_dot_IN.components
        for thisComponent in fixation_dot_IN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_dot_IN" ---
        thisExp.currentRoutine = fixation_dot_IN
        fixation_dot_IN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisIN_block, 'status') and thisIN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dot_IN* updates
            
            # if dot_IN is starting this frame...
            if dot_IN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot_IN.frameNStart = frameN  # exact frame index
                dot_IN.tStart = t  # local t and not account for scr refresh
                dot_IN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot_IN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot_IN.started')
                # update status
                dot_IN.status = STARTED
                dot_IN.setAutoDraw(True)
            
            # if dot_IN is active this frame...
            if dot_IN.status == STARTED:
                # update params
                pass
            
            # if dot_IN is stopping this frame...
            if dot_IN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot_IN.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    dot_IN.tStop = t  # not accounting for scr refresh
                    dot_IN.tStopRefresh = tThisFlipGlobal  # on global time
                    dot_IN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot_IN.stopped')
                    # update status
                    dot_IN.status = FINISHED
                    dot_IN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation_dot_IN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_dot_IN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_dot_IN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_dot_IN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_dot_IN" ---
        for thisComponent in fixation_dot_IN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_dot_IN
        fixation_dot_IN.tStop = globalClock.getTime(format='float')
        fixation_dot_IN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_dot_IN.stopped', fixation_dot_IN.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_dot_IN.maxDurationReached:
            routineTimer.addTime(-fixation_dot_IN.maxDuration)
        elif fixation_dot_IN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "pic_stim_IN" ---
        # create an object to store info about Routine pic_stim_IN
        pic_stim_IN = data.Routine(
            name='pic_stim_IN',
            components=[language_cue_IN, picture_stimuli_IN],
        )
        pic_stim_IN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        language_cue_IN.setImage(path_lc_IN)
        picture_stimuli_IN.setImage(path_ps_IN)
        # store start times for pic_stim_IN
        pic_stim_IN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pic_stim_IN.tStart = globalClock.getTime(format='float')
        pic_stim_IN.status = STARTED
        thisExp.addData('pic_stim_IN.started', pic_stim_IN.tStart)
        pic_stim_IN.maxDuration = None
        # keep track of which components have finished
        pic_stim_INComponents = pic_stim_IN.components
        for thisComponent in pic_stim_IN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pic_stim_IN" ---
        thisExp.currentRoutine = pic_stim_IN
        pic_stim_IN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisIN_block, 'status') and thisIN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *language_cue_IN* updates
            
            # if language_cue_IN is starting this frame...
            if language_cue_IN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                language_cue_IN.frameNStart = frameN  # exact frame index
                language_cue_IN.tStart = t  # local t and not account for scr refresh
                language_cue_IN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(language_cue_IN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'language_cue_IN.started')
                # update status
                language_cue_IN.status = STARTED
                language_cue_IN.setAutoDraw(True)
            
            # if language_cue_IN is active this frame...
            if language_cue_IN.status == STARTED:
                # update params
                pass
            
            # if language_cue_IN is stopping this frame...
            if language_cue_IN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > language_cue_IN.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    language_cue_IN.tStop = t  # not accounting for scr refresh
                    language_cue_IN.tStopRefresh = tThisFlipGlobal  # on global time
                    language_cue_IN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'language_cue_IN.stopped')
                    # update status
                    language_cue_IN.status = FINISHED
                    language_cue_IN.setAutoDraw(False)
            
            # *picture_stimuli_IN* updates
            
            # if picture_stimuli_IN is starting this frame...
            if picture_stimuli_IN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                picture_stimuli_IN.frameNStart = frameN  # exact frame index
                picture_stimuli_IN.tStart = t  # local t and not account for scr refresh
                picture_stimuli_IN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(picture_stimuli_IN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'picture_stimuli_IN.started')
                # update status
                picture_stimuli_IN.status = STARTED
                picture_stimuli_IN.setAutoDraw(True)
            
            # if picture_stimuli_IN is active this frame...
            if picture_stimuli_IN.status == STARTED:
                # update params
                pass
            
            # if picture_stimuli_IN is stopping this frame...
            if picture_stimuli_IN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > picture_stimuli_IN.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    picture_stimuli_IN.tStop = t  # not accounting for scr refresh
                    picture_stimuli_IN.tStopRefresh = tThisFlipGlobal  # on global time
                    picture_stimuli_IN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'picture_stimuli_IN.stopped')
                    # update status
                    picture_stimuli_IN.status = FINISHED
                    picture_stimuli_IN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=pic_stim_IN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                pic_stim_IN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if pic_stim_IN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in pic_stim_IN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pic_stim_IN" ---
        for thisComponent in pic_stim_IN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pic_stim_IN
        pic_stim_IN.tStop = globalClock.getTime(format='float')
        pic_stim_IN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pic_stim_IN.stopped', pic_stim_IN.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if pic_stim_IN.maxDurationReached:
            routineTimer.addTime(-pic_stim_IN.maxDuration)
        elif pic_stim_IN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "blank_screen_IN" ---
        # create an object to store info about Routine blank_screen_IN
        blank_screen_IN = data.Routine(
            name='blank_screen_IN',
            components=[blank_IN],
        )
        blank_screen_IN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank_screen_IN
        blank_screen_IN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank_screen_IN.tStart = globalClock.getTime(format='float')
        blank_screen_IN.status = STARTED
        thisExp.addData('blank_screen_IN.started', blank_screen_IN.tStart)
        blank_screen_IN.maxDuration = None
        # keep track of which components have finished
        blank_screen_INComponents = blank_screen_IN.components
        for thisComponent in blank_screen_IN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_screen_IN" ---
        thisExp.currentRoutine = blank_screen_IN
        blank_screen_IN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisIN_block, 'status') and thisIN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blank_IN* updates
            
            # if blank_IN is starting this frame...
            if blank_IN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blank_IN.frameNStart = frameN  # exact frame index
                blank_IN.tStart = t  # local t and not account for scr refresh
                blank_IN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blank_IN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank_IN.started')
                # update status
                blank_IN.status = STARTED
                blank_IN.setAutoDraw(True)
            
            # if blank_IN is active this frame...
            if blank_IN.status == STARTED:
                # update params
                pass
            
            # if blank_IN is stopping this frame...
            if blank_IN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank_IN.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    blank_IN.tStop = t  # not accounting for scr refresh
                    blank_IN.tStopRefresh = tThisFlipGlobal  # on global time
                    blank_IN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_IN.stopped')
                    # update status
                    blank_IN.status = FINISHED
                    blank_IN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank_screen_IN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank_screen_IN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank_screen_IN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank_screen_IN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_screen_IN" ---
        for thisComponent in blank_screen_IN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank_screen_IN
        blank_screen_IN.tStop = globalClock.getTime(format='float')
        blank_screen_IN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank_screen_IN.stopped', blank_screen_IN.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank_screen_IN.maxDurationReached:
            routineTimer.addTime(-blank_screen_IN.maxDuration)
        elif blank_screen_IN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisIN_block as finished
        if hasattr(thisIN_block, 'status'):
            thisIN_block.status = FINISHED
        # if awaiting a pause, pause now
        if IN_block.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            IN_block.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'IN_block'
    IN_block.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    EN_block = data.TrialHandler2(
        name='EN_block',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('condition_EN.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(EN_block)  # add the loop to the experiment
    thisEN_block = EN_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEN_block.rgb)
    if thisEN_block != None:
        for paramName in thisEN_block:
            globals()[paramName] = thisEN_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisEN_block in EN_block:
        EN_block.status = STARTED
        if hasattr(thisEN_block, 'status'):
            thisEN_block.status = STARTED
        currentLoop = EN_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisEN_block.rgb)
        if thisEN_block != None:
            for paramName in thisEN_block:
                globals()[paramName] = thisEN_block[paramName]
        
        # --- Prepare to start Routine "fixation_dot_EN" ---
        # create an object to store info about Routine fixation_dot_EN
        fixation_dot_EN = data.Routine(
            name='fixation_dot_EN',
            components=[dot_EN],
        )
        fixation_dot_EN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation_dot_EN
        fixation_dot_EN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_dot_EN.tStart = globalClock.getTime(format='float')
        fixation_dot_EN.status = STARTED
        thisExp.addData('fixation_dot_EN.started', fixation_dot_EN.tStart)
        fixation_dot_EN.maxDuration = None
        # keep track of which components have finished
        fixation_dot_ENComponents = fixation_dot_EN.components
        for thisComponent in fixation_dot_EN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_dot_EN" ---
        thisExp.currentRoutine = fixation_dot_EN
        fixation_dot_EN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisEN_block, 'status') and thisEN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dot_EN* updates
            
            # if dot_EN is starting this frame...
            if dot_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot_EN.frameNStart = frameN  # exact frame index
                dot_EN.tStart = t  # local t and not account for scr refresh
                dot_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot_EN.started')
                # update status
                dot_EN.status = STARTED
                dot_EN.setAutoDraw(True)
            
            # if dot_EN is active this frame...
            if dot_EN.status == STARTED:
                # update params
                pass
            
            # if dot_EN is stopping this frame...
            if dot_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot_EN.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    dot_EN.tStop = t  # not accounting for scr refresh
                    dot_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    dot_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot_EN.stopped')
                    # update status
                    dot_EN.status = FINISHED
                    dot_EN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation_dot_EN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_dot_EN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_dot_EN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_dot_EN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_dot_EN" ---
        for thisComponent in fixation_dot_EN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_dot_EN
        fixation_dot_EN.tStop = globalClock.getTime(format='float')
        fixation_dot_EN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_dot_EN.stopped', fixation_dot_EN.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_dot_EN.maxDurationReached:
            routineTimer.addTime(-fixation_dot_EN.maxDuration)
        elif fixation_dot_EN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "pic_stim_EN" ---
        # create an object to store info about Routine pic_stim_EN
        pic_stim_EN = data.Routine(
            name='pic_stim_EN',
            components=[language_cue_EN, picture_stimuli_EN],
        )
        pic_stim_EN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        language_cue_EN.setImage(path_lc_EN)
        picture_stimuli_EN.setImage(path_ps_EN)
        # store start times for pic_stim_EN
        pic_stim_EN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pic_stim_EN.tStart = globalClock.getTime(format='float')
        pic_stim_EN.status = STARTED
        thisExp.addData('pic_stim_EN.started', pic_stim_EN.tStart)
        pic_stim_EN.maxDuration = None
        # keep track of which components have finished
        pic_stim_ENComponents = pic_stim_EN.components
        for thisComponent in pic_stim_EN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pic_stim_EN" ---
        thisExp.currentRoutine = pic_stim_EN
        pic_stim_EN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisEN_block, 'status') and thisEN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *language_cue_EN* updates
            
            # if language_cue_EN is starting this frame...
            if language_cue_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                language_cue_EN.frameNStart = frameN  # exact frame index
                language_cue_EN.tStart = t  # local t and not account for scr refresh
                language_cue_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(language_cue_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'language_cue_EN.started')
                # update status
                language_cue_EN.status = STARTED
                language_cue_EN.setAutoDraw(True)
            
            # if language_cue_EN is active this frame...
            if language_cue_EN.status == STARTED:
                # update params
                pass
            
            # if language_cue_EN is stopping this frame...
            if language_cue_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > language_cue_EN.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    language_cue_EN.tStop = t  # not accounting for scr refresh
                    language_cue_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    language_cue_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'language_cue_EN.stopped')
                    # update status
                    language_cue_EN.status = FINISHED
                    language_cue_EN.setAutoDraw(False)
            
            # *picture_stimuli_EN* updates
            
            # if picture_stimuli_EN is starting this frame...
            if picture_stimuli_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                picture_stimuli_EN.frameNStart = frameN  # exact frame index
                picture_stimuli_EN.tStart = t  # local t and not account for scr refresh
                picture_stimuli_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(picture_stimuli_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'picture_stimuli_EN.started')
                # update status
                picture_stimuli_EN.status = STARTED
                picture_stimuli_EN.setAutoDraw(True)
            
            # if picture_stimuli_EN is active this frame...
            if picture_stimuli_EN.status == STARTED:
                # update params
                pass
            
            # if picture_stimuli_EN is stopping this frame...
            if picture_stimuli_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > picture_stimuli_EN.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    picture_stimuli_EN.tStop = t  # not accounting for scr refresh
                    picture_stimuli_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    picture_stimuli_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'picture_stimuli_EN.stopped')
                    # update status
                    picture_stimuli_EN.status = FINISHED
                    picture_stimuli_EN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=pic_stim_EN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                pic_stim_EN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if pic_stim_EN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in pic_stim_EN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pic_stim_EN" ---
        for thisComponent in pic_stim_EN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pic_stim_EN
        pic_stim_EN.tStop = globalClock.getTime(format='float')
        pic_stim_EN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pic_stim_EN.stopped', pic_stim_EN.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if pic_stim_EN.maxDurationReached:
            routineTimer.addTime(-pic_stim_EN.maxDuration)
        elif pic_stim_EN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "blank_screen_EN" ---
        # create an object to store info about Routine blank_screen_EN
        blank_screen_EN = data.Routine(
            name='blank_screen_EN',
            components=[blank_EN],
        )
        blank_screen_EN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank_screen_EN
        blank_screen_EN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank_screen_EN.tStart = globalClock.getTime(format='float')
        blank_screen_EN.status = STARTED
        thisExp.addData('blank_screen_EN.started', blank_screen_EN.tStart)
        blank_screen_EN.maxDuration = None
        # keep track of which components have finished
        blank_screen_ENComponents = blank_screen_EN.components
        for thisComponent in blank_screen_EN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_screen_EN" ---
        thisExp.currentRoutine = blank_screen_EN
        blank_screen_EN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisEN_block, 'status') and thisEN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blank_EN* updates
            
            # if blank_EN is starting this frame...
            if blank_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blank_EN.frameNStart = frameN  # exact frame index
                blank_EN.tStart = t  # local t and not account for scr refresh
                blank_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blank_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank_EN.started')
                # update status
                blank_EN.status = STARTED
                blank_EN.setAutoDraw(True)
            
            # if blank_EN is active this frame...
            if blank_EN.status == STARTED:
                # update params
                pass
            
            # if blank_EN is stopping this frame...
            if blank_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank_EN.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    blank_EN.tStop = t  # not accounting for scr refresh
                    blank_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    blank_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_EN.stopped')
                    # update status
                    blank_EN.status = FINISHED
                    blank_EN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank_screen_EN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank_screen_EN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank_screen_EN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank_screen_EN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_screen_EN" ---
        for thisComponent in blank_screen_EN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank_screen_EN
        blank_screen_EN.tStop = globalClock.getTime(format='float')
        blank_screen_EN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank_screen_EN.stopped', blank_screen_EN.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank_screen_EN.maxDurationReached:
            routineTimer.addTime(-blank_screen_EN.maxDuration)
        elif blank_screen_EN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisEN_block as finished
        if hasattr(thisEN_block, 'status'):
            thisEN_block.status = FINISHED
        # if awaiting a pause, pause now
        if EN_block.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            EN_block.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'EN_block'
    EN_block.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    mixed_block = data.TrialHandler2(
        name='mixed_block',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('condition.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(mixed_block)  # add the loop to the experiment
    thisMixed_block = mixed_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMixed_block.rgb)
    if thisMixed_block != None:
        for paramName in thisMixed_block:
            globals()[paramName] = thisMixed_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisMixed_block in mixed_block:
        mixed_block.status = STARTED
        if hasattr(thisMixed_block, 'status'):
            thisMixed_block.status = STARTED
        currentLoop = mixed_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisMixed_block.rgb)
        if thisMixed_block != None:
            for paramName in thisMixed_block:
                globals()[paramName] = thisMixed_block[paramName]
        
        # --- Prepare to start Routine "fixation_dot" ---
        # create an object to store info about Routine fixation_dot
        fixation_dot = data.Routine(
            name='fixation_dot',
            components=[dot],
        )
        fixation_dot.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation_dot
        fixation_dot.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_dot.tStart = globalClock.getTime(format='float')
        fixation_dot.status = STARTED
        thisExp.addData('fixation_dot.started', fixation_dot.tStart)
        fixation_dot.maxDuration = None
        # keep track of which components have finished
        fixation_dotComponents = fixation_dot.components
        for thisComponent in fixation_dot.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_dot" ---
        thisExp.currentRoutine = fixation_dot
        fixation_dot.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisMixed_block, 'status') and thisMixed_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dot* updates
            
            # if dot is starting this frame...
            if dot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot.frameNStart = frameN  # exact frame index
                dot.tStart = t  # local t and not account for scr refresh
                dot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot.started')
                # update status
                dot.status = STARTED
                dot.setAutoDraw(True)
            
            # if dot is active this frame...
            if dot.status == STARTED:
                # update params
                pass
            
            # if dot is stopping this frame...
            if dot.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    dot.tStop = t  # not accounting for scr refresh
                    dot.tStopRefresh = tThisFlipGlobal  # on global time
                    dot.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot.stopped')
                    # update status
                    dot.status = FINISHED
                    dot.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation_dot,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_dot.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_dot.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_dot.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_dot" ---
        for thisComponent in fixation_dot.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_dot
        fixation_dot.tStop = globalClock.getTime(format='float')
        fixation_dot.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_dot.stopped', fixation_dot.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_dot.maxDurationReached:
            routineTimer.addTime(-fixation_dot.maxDuration)
        elif fixation_dot.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "pic_stim_lang_cue" ---
        # create an object to store info about Routine pic_stim_lang_cue
        pic_stim_lang_cue = data.Routine(
            name='pic_stim_lang_cue',
            components=[language_cue, picture_stimuli],
        )
        pic_stim_lang_cue.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        language_cue.setImage(path_lc)
        picture_stimuli.setImage(path_ps)
        # store start times for pic_stim_lang_cue
        pic_stim_lang_cue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pic_stim_lang_cue.tStart = globalClock.getTime(format='float')
        pic_stim_lang_cue.status = STARTED
        thisExp.addData('pic_stim_lang_cue.started', pic_stim_lang_cue.tStart)
        pic_stim_lang_cue.maxDuration = None
        # keep track of which components have finished
        pic_stim_lang_cueComponents = pic_stim_lang_cue.components
        for thisComponent in pic_stim_lang_cue.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pic_stim_lang_cue" ---
        thisExp.currentRoutine = pic_stim_lang_cue
        pic_stim_lang_cue.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisMixed_block, 'status') and thisMixed_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *language_cue* updates
            
            # if language_cue is starting this frame...
            if language_cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                language_cue.frameNStart = frameN  # exact frame index
                language_cue.tStart = t  # local t and not account for scr refresh
                language_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(language_cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'language_cue.started')
                # update status
                language_cue.status = STARTED
                language_cue.setAutoDraw(True)
            
            # if language_cue is active this frame...
            if language_cue.status == STARTED:
                # update params
                pass
            
            # if language_cue is stopping this frame...
            if language_cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > language_cue.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    language_cue.tStop = t  # not accounting for scr refresh
                    language_cue.tStopRefresh = tThisFlipGlobal  # on global time
                    language_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'language_cue.stopped')
                    # update status
                    language_cue.status = FINISHED
                    language_cue.setAutoDraw(False)
            
            # *picture_stimuli* updates
            
            # if picture_stimuli is starting this frame...
            if picture_stimuli.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                picture_stimuli.frameNStart = frameN  # exact frame index
                picture_stimuli.tStart = t  # local t and not account for scr refresh
                picture_stimuli.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(picture_stimuli, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'picture_stimuli.started')
                # update status
                picture_stimuli.status = STARTED
                picture_stimuli.setAutoDraw(True)
            
            # if picture_stimuli is active this frame...
            if picture_stimuli.status == STARTED:
                # update params
                pass
            
            # if picture_stimuli is stopping this frame...
            if picture_stimuli.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > picture_stimuli.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    picture_stimuli.tStop = t  # not accounting for scr refresh
                    picture_stimuli.tStopRefresh = tThisFlipGlobal  # on global time
                    picture_stimuli.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'picture_stimuli.stopped')
                    # update status
                    picture_stimuli.status = FINISHED
                    picture_stimuli.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=pic_stim_lang_cue,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                pic_stim_lang_cue.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if pic_stim_lang_cue.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in pic_stim_lang_cue.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pic_stim_lang_cue" ---
        for thisComponent in pic_stim_lang_cue.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pic_stim_lang_cue
        pic_stim_lang_cue.tStop = globalClock.getTime(format='float')
        pic_stim_lang_cue.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pic_stim_lang_cue.stopped', pic_stim_lang_cue.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if pic_stim_lang_cue.maxDurationReached:
            routineTimer.addTime(-pic_stim_lang_cue.maxDuration)
        elif pic_stim_lang_cue.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "blank_screen_2" ---
        # create an object to store info about Routine blank_screen_2
        blank_screen_2 = data.Routine(
            name='blank_screen_2',
            components=[blank],
        )
        blank_screen_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank_screen_2
        blank_screen_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank_screen_2.tStart = globalClock.getTime(format='float')
        blank_screen_2.status = STARTED
        thisExp.addData('blank_screen_2.started', blank_screen_2.tStart)
        blank_screen_2.maxDuration = None
        # keep track of which components have finished
        blank_screen_2Components = blank_screen_2.components
        for thisComponent in blank_screen_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_screen_2" ---
        thisExp.currentRoutine = blank_screen_2
        blank_screen_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisMixed_block, 'status') and thisMixed_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blank* updates
            
            # if blank is starting this frame...
            if blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blank.frameNStart = frameN  # exact frame index
                blank.tStart = t  # local t and not account for scr refresh
                blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank.started')
                # update status
                blank.status = STARTED
                blank.setAutoDraw(True)
            
            # if blank is active this frame...
            if blank.status == STARTED:
                # update params
                pass
            
            # if blank is stopping this frame...
            if blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    blank.tStop = t  # not accounting for scr refresh
                    blank.tStopRefresh = tThisFlipGlobal  # on global time
                    blank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank.stopped')
                    # update status
                    blank.status = FINISHED
                    blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank_screen_2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank_screen_2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank_screen_2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank_screen_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_screen_2" ---
        for thisComponent in blank_screen_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank_screen_2
        blank_screen_2.tStop = globalClock.getTime(format='float')
        blank_screen_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank_screen_2.stopped', blank_screen_2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank_screen_2.maxDurationReached:
            routineTimer.addTime(-blank_screen_2.maxDuration)
        elif blank_screen_2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisMixed_block as finished
        if hasattr(thisMixed_block, 'status'):
            thisMixed_block.status = FINISHED
        # if awaiting a pause, pause now
        if mixed_block.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            mixed_block.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'mixed_block'
    mixed_block.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "non_switching_block" ---
    # create an object to store info about Routine non_switching_block
    non_switching_block = data.Routine(
        name='non_switching_block',
        components=[text_non_switching],
    )
    non_switching_block.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code
    breakDuration = 30
    countdown = breakDuration
    # store start times for non_switching_block
    non_switching_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    non_switching_block.tStart = globalClock.getTime(format='float')
    non_switching_block.status = STARTED
    thisExp.addData('non_switching_block.started', non_switching_block.tStart)
    non_switching_block.maxDuration = None
    # keep track of which components have finished
    non_switching_blockComponents = non_switching_block.components
    for thisComponent in non_switching_block.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "non_switching_block" ---
    thisExp.currentRoutine = non_switching_block
    non_switching_block.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_non_switching* updates
        
        # if text_non_switching is starting this frame...
        if text_non_switching.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_non_switching.frameNStart = frameN  # exact frame index
            text_non_switching.tStart = t  # local t and not account for scr refresh
            text_non_switching.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_non_switching, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_non_switching.started')
            # update status
            text_non_switching.status = STARTED
            text_non_switching.setAutoDraw(True)
        
        # if text_non_switching is active this frame...
        if text_non_switching.status == STARTED:
            # update params
            text_non_switching.setText('Jeda istirahat dalam $countdown detik', log=False)
        # Run 'Each Frame' code from code
        remaining = int(breakDuration - t)
        countdown = max(remaining, 0)
        
        if t >= breakDuration:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=non_switching_block,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            non_switching_block.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if non_switching_block.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in non_switching_block.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "non_switching_block" ---
    for thisComponent in non_switching_block.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for non_switching_block
    non_switching_block.tStop = globalClock.getTime(format='float')
    non_switching_block.tStopRefresh = tThisFlipGlobal
    thisExp.addData('non_switching_block.stopped', non_switching_block.tStop)
    thisExp.nextEntry()
    # the Routine "non_switching_block" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
