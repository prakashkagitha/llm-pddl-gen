(define (problem blocksworld-p06)
  (:domain blocksworld)
  (:objects block1 block2 block3 block4 block5 block6 block7 block8 block9 block10 block11 block12 block13 block14 )
  (:init 
    (on-table block3)
    (on block9 block3)
    (on block4 block9)
    (on block2 block4)
    (on block14 block2)
    (on block12 block14)
    (on block7 block12)
    (clear block7)
    (on-table block11)
    (on block8 block11)
    (on block1 block8)
    (on block6 block1)
    (on block10 block6)
    (on block13 block10)
    (clear block13)
    (on-table block5)
    (clear block5)
    (arm-empty)
  )
  (:goal (and 
    (on-table block6)
    (on block8 block6)
    (on-table block5)
    (on-table block12)
    (on-table block1)
    (on-table block11)
    (on block14 block11)
    (on block7 block14)
    (on-table block4)
    (on-table block9)
    (on block3 block9)
    (on-table block13)
    (on-table block2)
    (on-table block10)
  ))
)